import os
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer
from vllm import LLM

from pra.utils.types import Action, Config
from pra.utils.args import get_base_parser, add_beam_args, parse_with_config
from pra.utils.trace_beam import BeamTrace
from pra.utils.stages import run_reason_beam
from pra.utils.stages_beam import run_reward_beam, run_search_beam
from pra.utils.scoring import save_results, format_live_accuracy, print_beam_accuracy_comparison
from pra.utils.retriever import get_retriever


def parse_args(argv=None):
    parser = get_base_parser()
    parser = add_beam_args(parser)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split dataset into N shards (by global index modulo).")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Which shard to run (0 <= shard_id < num_shards).")
    return parse_with_config(
        parser,
        argv,
        required=(
            "input_jsonl_path",
            "policy_model_path",
            "trace_dir",
            "reward_model_path",
            "visible_gpus",
            "tensor_parallel_size",
            "retriever_device",
            "retriever_index_path",
            "retriever_topk",
            "retriever_rerank_topk",
            "retriever_sources",
            "rerank_workers",
            "max_docs",
            "sample_size",
            "batch_size_pi_theta",
            "batch_size_mu_phi",
            "batch_size_rho",
            "max_workers",
            "theta_qual",
            "theta_dep",
            "repetition_penalty",
            "pi_theta_max_tokens",
            "pi_theta_temperature",
            "pi_theta_top_p",
            "pi_theta_top_k",
            "beam_width",
            "beam_candidates",
            "beam_final_selection",
        ),
    )


def _resolve_reward_model_url(args) -> str:
    """
    Priority:
      1) --reward_model_url (already present on args)
      2) $VLLM_PORT
      3) sharded run: $VLLM_PORT_BASE (default 8400) + shard_id
      4) non-sharded SLURM run: $VLLM_PORT_BASE + (SLURM_JOB_ID % 1000)
      5) non-sharded local run: $VLLM_PORT_BASE
    """
    if args.reward_model_url:
        return args.reward_model_url

    base_port = int(os.environ.get("VLLM_PORT_BASE", "8400"))
    vllm_port = os.environ.get("VLLM_PORT")
    if vllm_port:
        port = int(vllm_port)
    elif args.num_shards > 1:
        port = base_port + args.shard_id
    else:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id is not None and slurm_job_id.isdigit():
            port = base_port + (int(slurm_job_id) % 1000)
        else:
            port = base_port
    return f"http://localhost:{port}/v1/completions"


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main(args):
    _configure_gpu(args)
    indexed = _load_data(args)
    os.makedirs(args.trace_dir, exist_ok=True)

    retriever_sources = _prepare_retriever(args)
    pi_theta, pi_theta_tokenizer, mu_phi_tokenizer = _load_models(args)
    config = _build_config(args)
    traces = (BeamTrace(j, d, pi_theta_tokenizer, config) for j, d in indexed)

    completed = _run_beam_search(
        traces, pi_theta, mu_phi_tokenizer, config, retriever_sources, args
    )
    _save_results(completed, args)


def _configure_gpu(args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        vis = os.environ["CUDA_VISIBLE_DEVICES"]
        if args.visible_gpus and args.visible_gpus != vis:
            print(
                f"[INFO] CUDA_VISIBLE_DEVICES already set ({vis}); "
                f"ignoring config visible_gpus={args.visible_gpus!r}"
            )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus


def _load_data(args) -> list[tuple[int, dict]]:
    """Load and shard examples, preserving their original indices."""
    with open(args.input_jsonl_path) as f:
        all_data = [json.loads(line) for line in f]
    if args.sample_size is not None:
        all_data = all_data[: args.sample_size]

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    # Keep original global indices as `j` so shards can be merged safely.
    indexed = list(enumerate(all_data))
    if args.num_shards > 1:
        indexed = [(j, d) for j, d in indexed if (j % args.num_shards) == args.shard_id]

    print(f"[INFO] Sharding: num_shards={args.num_shards}, shard_id={args.shard_id}")
    print(f"[INFO] Loaded {len(all_data)} total examples, running {len(indexed)} on this shard.")
    return indexed


def _prepare_retriever(args) -> list[str]:
    retriever_sources = [s.strip() for s in args.retriever_sources.split(",")]
    print(f"[INFO] Pre-initializing retriever on {args.retriever_device}...")
    print(f"[INFO] Retriever sources: {retriever_sources}")
    get_retriever(
        device=args.retriever_device,
        rerank_devices=[args.retriever_device],
        index_path=args.retriever_index_path,
        sources=retriever_sources,
    )
    return retriever_sources


def _load_models(args):
    """Load the policy model and both tokenizers for beam inference."""
    print(f"[INFO] pi_theta: {args.policy_model_path}")
    print(f"[INFO] mu_phi: {args.reward_model_path}")
    args.reward_model_url = _resolve_reward_model_url(args)
    print(f"[INFO] reward_model_url: {args.reward_model_url}")
    print(f"[INFO] mode: beam (width={args.beam_width}, candidates={args.beam_candidates})")

    mu_phi_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    pi_theta = LLM(
        model=args.policy_model_path,
        dtype="bfloat16",
        tokenizer=args.policy_model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    pi_theta_tokenizer = pi_theta.get_tokenizer()
    print(f"[INFO] prefix_caching: {args.enable_prefix_caching}")
    return pi_theta, pi_theta_tokenizer, mu_phi_tokenizer


def _build_config(args) -> Config:
    return Config(
        theta_qual=args.theta_qual,
        theta_dep=args.theta_dep,
        K_max=args.K_max,
        max_docs=args.max_docs,
        aggregate_steps_for_reward=args.aggregate_steps_for_reward,
        beam_width=args.beam_width,
        beam_candidates=args.beam_candidates,
        beam_final_selection=args.beam_final_selection,
        beam_require_parsable_answer=args.beam_require_parsable_answer,
        adaptive_beam=args.adaptive_beam,
        beam_candidates_start=args.beam_candidates_start,
        beam_candidates_end=args.beam_candidates_end,
        beam_width_start=args.beam_width_start,
        beam_width_end=args.beam_width_end,
        beam_schedule_power=args.beam_schedule_power if args.beam_schedule_power is not None else 1.0,
        retriever_use_all_steps=args.retriever_use_all_steps,
    )


def _run_beam_search(
    traces, pi_theta, mu_phi_tokenizer, config: Config, retriever_sources: list[str], args
) -> list[BeamTrace]:
    """Schedule reasoning, reward, and search stages until all traces finish."""
    uncompleted = deque(traces)
    completed: list = []
    step = 0

    batch_sizes = {
        Action.REASON: args.batch_size_pi_theta,
        Action.REWARD: args.batch_size_mu_phi,
    }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while uncompleted:
            step += 1

            buckets = {Action.REASON: [], Action.REWARD: [], Action.SEARCH: []}
            while uncompleted:
                t = uncompleted.popleft()
                if t.action in buckets:
                    buckets[t.action].append(t)
                else:
                    completed.append(t)

            step_counts: dict[int, int] = {}
            for t in buckets[Action.REASON] + buckets[Action.REWARD] + buckets[Action.SEARCH]:
                step_counts[t.i] = step_counts.get(t.i, 0) + 1
            step_dist = " ".join(f"{k}:{v}" for k, v in sorted(step_counts.items()))

            total_beams = sum(len(t.get_active_beams()) for t in buckets[Action.REASON])
            batch_candidates = sum(len(t.pending_candidates) for t in buckets[Action.REWARD])
            k_max_str = "unlimited" if config.K_max is None else config.K_max

            print(
                f"\n[Iter {step}] Steps[{step_dist}] K_max={k_max_str} | "
                f"REASON:{len(buckets[Action.REASON])} (beams:{total_beams}) "
                f"REWARD:{len(buckets[Action.REWARD])} (candidates:{batch_candidates}) "
                f"SEARCH:{len(buckets[Action.SEARCH])}"
            )

            if buckets[Action.SEARCH]:
                updated, done = run_search_beam(buckets[Action.SEARCH], retriever_sources, args)
                uncompleted.extend(updated)
                completed.extend(done)

            futures = {}
            for action, batch_size in batch_sizes.items():
                runner = run_reason_beam if action == Action.REASON else run_reward_beam
                model_or_tokenizer = pi_theta if action == Action.REASON else mu_phi_tokenizer
                for chunk in _chunks(buckets[action], batch_size):
                    if chunk:
                        futures[executor.submit(runner, chunk, model_or_tokenizer, config, args)] = action

            if not futures and not buckets[Action.SEARCH]:
                break

            for future in as_completed(futures):
                updated, done = future.result()
                uncompleted.extend(updated)
                completed.extend(done)

            print(
                f"         Remaining: {len(uncompleted)}, Completed: {len(completed)} | "
                f"{format_live_accuracy(completed)}"
            )

    return completed


def _save_results(completed: list[BeamTrace], args) -> None:
    config_name = args.config_name
    if args.num_shards > 1:
        base = config_name or "beam"
        config_name = f"{base}_sh{args.shard_id}of{args.num_shards}"

    output_dir = save_results(completed, args.trace_dir, config_name, "beam", args)
    print_beam_accuracy_comparison(completed)
    print(f"\n[DONE] Saved to: {output_dir}")


def main_cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    main_cli()
