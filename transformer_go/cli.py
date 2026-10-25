import argparse
import json
from .config import load_config
from .runtime import device_report, seed_all, select_device, select_inference_device
from .model import PolicyValueNet
from .search import Evaluator
from .selfplay import generate
from .checkpoint import load_checkpoint, model_version
from .train import train
from ._native import Board
from .gating import DEFAULT_GATING_GAMES, DEFAULT_REQUIRED_WIN_PROP
from .inference import validate_batching


def report(record):
    print(json.dumps(record), flush=True)


def cmd_doctor(_):
    print(json.dumps(device_report(), indent=2))


def evaluator_options(args):
    if args.inference_backend == "torch":
        return {}
    return {"backend": args.inference_backend, "coreml_batch_size": args.coreml_batch_size,
            "coreml_cache_dir": args.coreml_cache_dir}


def cmd_selfplay(args):
    config = load_config(args.config)
    seed_all(config.seed)
    if args.workers < 1:
        raise ValueError("workers must be a positive integer")
    validate_batching(args.inference_batch_size, args.inference_wait_ms)
    device = select_inference_device(config.device, args.inference_backend)
    if args.workers > 1:
        from .actors import generate_parallel
        paths = generate_parallel(config, args.checkpoint, args.output, args.games, args.workers,
                                  args.seed, min_moves=args.min_moves, score_utility=args.score_utility,
                                  report=report, device=device,
                                  inference_batch_size=args.inference_batch_size,
                                  inference_wait_ms=args.inference_wait_ms, selfplay_policy=args.selfplay_policy,
                                  evaluator_options=evaluator_options(args))
        report({"event": "parallel_selfplay_complete", "paths": paths})
        return
    model = PolicyValueNet(config.model).to(device)
    if args.checkpoint:
        model, config, _ = load_checkpoint(args.checkpoint, device, selfplay_only=True)
        version = model_version(args.checkpoint)
    else:
        version = "untrained"
    evaluator = Evaluator(model, device, **evaluator_options(args))
    generate(config, evaluator, args.output, args.games, args.seed, version, report,
             min_moves=args.min_moves, score_utility=args.score_utility, selfplay_policy=args.selfplay_policy)
    print(json.dumps({"event": "selfplay_complete", "evaluator_calls": evaluator.calls,
                      "positions": evaluator.positions,
                      "device": "coreml-cpu-and-ne" if evaluator.backend == "coreml" else str(device),
                      "inference_backend": evaluator.backend}))


def cmd_train(args):
    config = load_config(args.config)
    seed_all(config.seed)
    device = select_device(config.device)
    model = PolicyValueNet(config.model).to(device)
    resume = None
    if args.resume:
        model, loaded, resume = load_checkpoint(args.resume, device, selfplay_only=True)
        if loaded.to_dict() != config.to_dict():
            raise ValueError("resume checkpoint configuration differs from requested config")
    train(config, model, device, args.data, args.output, args.steps, resume, report,
          metrics_dir=args.metrics_dir)
    print(json.dumps({"event": "training_complete", "checkpoint": args.output, "device": str(device)}))


def cmd_gtp(args):
    config = load_config(args.config)
    seed_all(config.seed)
    device = select_inference_device(config.device, args.inference_backend)
    model = PolicyValueNet(config.model).to(device)
    if args.checkpoint:
        model, config, _ = load_checkpoint(args.checkpoint, device)
    from .gtp import GTP
    GTP(config, Evaluator(model, device, **evaluator_options(args))).run()


def cmd_eval(args):
    from .evaluate import evaluate
    config = load_config(args.config)
    if args.workers < 1:
        raise ValueError("workers must be a positive integer")
    validate_batching(args.inference_batch_size, args.inference_wait_ms)
    device = select_inference_device(config.device, args.inference_backend)
    recording = {"sgf_dir": args.sgf_dir, "temperature": args.temperature,
                 "candidate_name": f"{args.candidate} sha256={model_version(args.candidate)}",
                 "baseline_name": f"{args.baseline} sha256={model_version(args.baseline)}"}
    if args.workers > 1:
        from .evaluate import evaluate_parallel
        print(json.dumps(evaluate_parallel(
            config, args.candidate, args.baseline, args.games, args.workers, report,
            score_utility=args.score_utility,
            required_candidate_win_prop=args.required_candidate_win_prop,
            seed=args.seed, device=device, inference_batch_size=args.inference_batch_size,
            inference_wait_ms=args.inference_wait_ms, evaluator_options=evaluator_options(args),
            **recording), indent=2))
        return
    candidate, saved, _ = load_checkpoint(args.candidate, device)
    baseline, previous, _ = load_checkpoint(args.baseline, device)
    if saved.board_size != config.board_size or previous.board_size != config.board_size:
        raise ValueError("evaluation board size differs from checkpoints")
    print(json.dumps(evaluate(config, Evaluator(candidate, device, **evaluator_options(args)),
                              Evaluator(baseline, device, **evaluator_options(args)), args.games,
                              report, score_utility=args.score_utility,
                              required_candidate_win_prop=args.required_candidate_win_prop,
                              seed=args.seed, **recording), indent=2))


def cmd_loop(args):
    from .pipeline import run_pipeline
    config = load_config(args.config)
    seed_all(config.seed)
    run_pipeline(config, select_device(config.device), args.output, args.iterations,
                 args.games, args.steps, args.eval_games, report=report, min_moves=args.min_moves,
                 score_utility=args.score_utility, promote_threshold=args.required_candidate_win_prop,
                 selfplay_workers=args.selfplay_workers, eval_workers=args.eval_workers,
                 inference_batch_size=args.inference_batch_size,
                 inference_wait_ms=args.inference_wait_ms, selfplay_policy=args.selfplay_policy,
                 evaluator_options=evaluator_options(args))


def cmd_rpc(args):
    from .rpc import RPC
    config = load_config(args.config)
    device = select_inference_device(config.device, args.inference_backend)
    if args.checkpoint: model, config, _ = load_checkpoint(args.checkpoint, device)
    else:
        seed_all(config.seed)
        model = PolicyValueNet(config.model).to(device)
    RPC(config, Evaluator(model, device, **evaluator_options(args))).run()


def cmd_play(args):
    from .web import serve
    from pathlib import Path
    config = load_config(args.config)
    device = select_inference_device(config.device, args.inference_backend)
    serve(config, device, Path(__file__).resolve().parent.parent,
          checkpoint=args.checkpoint, host=args.host, port=args.port,
          evaluator_options=evaluator_options(args))


def cmd_benchmark(args):
    import time
    from .runtime import synchronize
    from .search import search
    config = load_config(args.config)
    device = select_inference_device(config.device, args.inference_backend)
    seed_all(config.seed)
    model = PolicyValueNet(config.model).to(device)
    if args.checkpoint:
        model, config, _ = load_checkpoint(args.checkpoint, device)
    evaluator = Evaluator(model, device, **evaluator_options(args))
    board = Board(config.board_size, config.komi, config.rules)
    evaluator([board])
    synchronize(device)
    start = time.perf_counter()
    result = search(board, evaluator, config.search, config.seed)
    synchronize(device)
    elapsed = time.perf_counter()-start
    record = {"device": "coreml-cpu-and-ne" if evaluator.backend == "coreml" else str(device),
                      "inference_backend": evaluator.backend,
                      "compute_units": "CPU_AND_NE" if evaluator.backend == "coreml" else str(device),
                      "coreml_batch_size": args.coreml_batch_size if evaluator.backend == "coreml" else None,
                      "seconds": elapsed, "simulations": result["simulations"],
                      "simulations_per_second": result["simulations"]/elapsed,
                      "evaluator_calls": evaluator.calls-1, "parameters": sum(p.numel() for p in model.parameters())}
    if args.coreml_compute_plan and evaluator.coreml is not None:
        record["coreml_compute_plan"] = evaluator.coreml.compute_plan_report(config.board_size)
    print(json.dumps(record, indent=2))


def add_backend_options(parser):
    parser.add_argument("--inference-backend", choices=("torch", "coreml"), default="torch",
                        help="network inference backend; coreml allows Apple CPU/Neural Engine (default: torch)")
    parser.add_argument("--coreml-batch-size", type=int, default=16,
                        help="fixed Core ML batch size; requests are padded/chunked (default: 16)")
    parser.add_argument("--coreml-cache-dir", default="runs/coreml",
                        help="cache converted Core ML packages by weights/shape (default: runs/coreml)")


def add_inference_options(parser):
    parser.add_argument("--inference-batch-size", type=int, default=256,
                        help="max positions per centralized forward with multiple workers (default: 256)")
    parser.add_argument("--inference-wait-ms", type=float, default=2.0,
                        help="max collection wait for cross-game batches in milliseconds (default: 2)")


def add_selfplay_policy_option(parser):
    parser.add_argument("--selfplay-policy", choices=("katago", "legacy"), default="katago",
                        help="self-play exploration/target profile; runtime only, default: katago")


def main():
    parser = argparse.ArgumentParser(prog="transformer-go")
    sub = parser.add_subparsers(required=True)
    doctor = sub.add_parser("doctor"); doctor.set_defaults(func=cmd_doctor)
    for name, func in (("selfplay", cmd_selfplay), ("train", cmd_train), ("gtp", cmd_gtp), ("json-rpc", cmd_rpc)):
        p = sub.add_parser(name); p.add_argument("--config", default="configs/tiny.yaml"); p.set_defaults(func=func)
        if name != "train":
            add_backend_options(p)
        if name == "selfplay":
            add_inference_options(p)
            add_selfplay_policy_option(p)
            p.add_argument("--output", default="data/replay"); p.add_argument("--games", type=int, default=1)
            p.add_argument("--seed", type=int); p.add_argument("--checkpoint")
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--min-moves", type=int, default=0,
                           help="disable self-play pass before this many plies unless no board move is legal (default: 0)")
            p.add_argument("--score-utility", type=float, default=0.05,
                           help="small score prediction contribution to MCTS utility (default: 0.05)")
        elif name == "train":
            p.add_argument("--data", default="data/replay"); p.add_argument("--output", default="runs/model.pt")
            p.add_argument("--steps", type=int, default=100); p.add_argument("--resume")
            p.add_argument("--metrics-dir", help="loss log directory (default: OUTPUT_PARENT/training/OUTPUT_STEM)")
        else: p.add_argument("--checkpoint")
    p = sub.add_parser("play", help="start the local human-vs-engine web UI")
    add_backend_options(p)
    p.set_defaults(func=cmd_play)
    p.add_argument("--config", default="configs/tiny.yaml",
                   help="fallback config; checkpoint config is used when a weight is selected")
    p.add_argument("--checkpoint", help="initial weight (defaults to the newest runs/**/best.pt)")
    p.add_argument("--host", default="127.0.0.1", help="listen address (default: localhost only)")
    p.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")
    p = sub.add_parser("eval"); p.set_defaults(func=cmd_eval)
    add_backend_options(p)
    add_inference_options(p)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--candidate", required=True); p.add_argument("--baseline", required=True)
    p.add_argument("--games", type=int, default=DEFAULT_GATING_GAMES)
    p.add_argument("--workers", type=int, default=1,
                   help="spawned evaluation workers; 1 keeps strict serial gatekeeper execution")
    p.add_argument("--seed", type=int, help="evaluation seed (default: config seed)")
    p.add_argument("--sgf-dir", default="runs/evaluation",
                   help="save counted games in a unique match subdirectory (default: runs/evaluation)")
    p.add_argument("--temperature", type=float,
                   help="diagnostic fixed temperature; 0 selects maximum MCTS visits, incompatible with gatekeeper promotion")
    p.add_argument("--required-candidate-win-prop", type=float,
                   help="enable gatekeeper acceptance and early stopping at this match-point fraction; omit to play all games")
    p.add_argument("--score-utility", type=float, default=0.05,
                   help="small score prediction contribution to MCTS utility (default: 0.05)")
    p = sub.add_parser("loop"); p.set_defaults(func=cmd_loop)
    add_backend_options(p)
    add_inference_options(p)
    add_selfplay_policy_option(p)
    p.add_argument("--config", default="configs/tiny.yaml"); p.add_argument("--output", default="runs/loop")
    p.add_argument("--iterations", type=int, default=1); p.add_argument("--games", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-games", type=int, default=DEFAULT_GATING_GAMES,
                   help="maximum games per gatekeeper match (default: 200); may stop once decided")
    p.add_argument("--required-candidate-win-prop", type=float, default=DEFAULT_REQUIRED_WIN_PROP,
                   help="required candidate match-point fraction; ties accepted (default: 0.5, like KataGo)")
    p.add_argument("--min-moves", type=int, default=0,
                   help="disable self-play pass before this many plies; does not constrain evaluation (default: 0)")
    p.add_argument("--score-utility", type=float, default=0.05,
                   help="small score prediction contribution to MCTS utility (default: 0.05)")
    p.add_argument("--selfplay-workers", type=int, default=1,
                   help="spawned self-play actors per iteration (default: 1)")
    p.add_argument("--eval-workers", type=int, default=1,
                   help="spawned evaluation workers per match (default: 1)")
    p = sub.add_parser("benchmark"); p.set_defaults(func=cmd_benchmark)
    add_backend_options(p)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--checkpoint", help="benchmark this checkpoint instead of random weights")
    p.add_argument("--coreml-compute-plan", action="store_true",
                   help="report Core ML static device preferences; does not measure NPU utilization")
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
