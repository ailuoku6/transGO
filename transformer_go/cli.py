import argparse
import json
from .config import load_config
from .runtime import device_report, seed_all, select_device
from .model import PolicyValueNet
from .search import Evaluator
from .selfplay import generate
from .checkpoint import load_checkpoint, model_version
from .train import train
from ._native import Board
from .gating import DEFAULT_GATING_GAMES, DEFAULT_REQUIRED_WIN_PROP


def report(record):
    print(json.dumps(record), flush=True)


def cmd_doctor(_):
    print(json.dumps(device_report(), indent=2))


def cmd_selfplay(args):
    config = load_config(args.config)
    seed_all(config.seed)
    if args.workers < 1:
        raise ValueError("workers must be a positive integer")
    device = select_device(config.device)
    if args.workers > 1:
        from .actors import generate_parallel
        paths = generate_parallel(config, args.checkpoint, args.output, args.games, args.workers,
                                  args.seed, min_moves=args.min_moves, score_utility=args.score_utility,
                                  report=report, device=device)
        report({"event": "parallel_selfplay_complete", "paths": paths})
        return
    model = PolicyValueNet(config.model).to(device)
    if args.checkpoint:
        model, config, _ = load_checkpoint(args.checkpoint, device, selfplay_only=True)
        version = model_version(args.checkpoint)
    else:
        version = "untrained"
    evaluator = Evaluator(model, device)
    generate(config, evaluator, args.output, args.games, args.seed, version, report,
             min_moves=args.min_moves, score_utility=args.score_utility)
    print(json.dumps({"event": "selfplay_complete", "evaluator_calls": evaluator.calls,
                      "positions": evaluator.positions, "device": str(device)}))


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
    train(config, model, device, args.data, args.output, args.steps, resume, report)
    print(json.dumps({"event": "training_complete", "checkpoint": args.output, "device": str(device)}))


def cmd_gtp(args):
    config = load_config(args.config)
    seed_all(config.seed)
    device = select_device(config.device)
    model = PolicyValueNet(config.model).to(device)
    if args.checkpoint:
        model, config, _ = load_checkpoint(args.checkpoint, device)
    from .gtp import GTP
    GTP(config, Evaluator(model, device)).run()


def cmd_eval(args):
    from .evaluate import evaluate
    config = load_config(args.config)
    if args.workers < 1:
        raise ValueError("workers must be a positive integer")
    device = select_device(config.device)
    if args.workers > 1:
        from .evaluate import evaluate_parallel
        print(json.dumps(evaluate_parallel(
            config, args.candidate, args.baseline, args.games, args.workers, report,
            score_utility=args.score_utility,
            required_candidate_win_prop=args.required_candidate_win_prop,
            seed=args.seed, device=device), indent=2))
        return
    candidate, saved, _ = load_checkpoint(args.candidate, device)
    baseline, previous, _ = load_checkpoint(args.baseline, device)
    if saved.board_size != config.board_size or previous.board_size != config.board_size:
        raise ValueError("evaluation board size differs from checkpoints")
    print(json.dumps(evaluate(config, Evaluator(candidate, device), Evaluator(baseline, device), args.games,
                              report, score_utility=args.score_utility,
                              required_candidate_win_prop=args.required_candidate_win_prop,
                              seed=args.seed), indent=2))


def cmd_loop(args):
    from .pipeline import run_pipeline
    config = load_config(args.config)
    seed_all(config.seed)
    run_pipeline(config, select_device(config.device), args.output, args.iterations,
                 args.games, args.steps, args.eval_games, report=report, min_moves=args.min_moves,
                 score_utility=args.score_utility, promote_threshold=args.required_candidate_win_prop,
                 selfplay_workers=args.selfplay_workers, eval_workers=args.eval_workers)


def cmd_rpc(args):
    from .rpc import RPC
    config = load_config(args.config)
    device = select_device(config.device)
    if args.checkpoint: model, config, _ = load_checkpoint(args.checkpoint, device)
    else:
        seed_all(config.seed)
        model = PolicyValueNet(config.model).to(device)
    RPC(config, Evaluator(model, device)).run()


def cmd_benchmark(args):
    import time
    from .runtime import synchronize
    from .search import search
    config = load_config(args.config)
    device = select_device(config.device)
    seed_all(config.seed)
    model = PolicyValueNet(config.model).to(device)
    evaluator = Evaluator(model, device)
    board = Board(config.board_size, config.komi, config.rules)
    evaluator([board])
    synchronize(device)
    start = time.perf_counter()
    result = search(board, evaluator, config.search, config.seed)
    synchronize(device)
    elapsed = time.perf_counter()-start
    print(json.dumps({"device": str(device), "seconds": elapsed, "simulations": result["simulations"],
                      "simulations_per_second": result["simulations"]/elapsed,
                      "evaluator_calls": evaluator.calls-1, "parameters": sum(p.numel() for p in model.parameters())}, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="transformer-go")
    sub = parser.add_subparsers(required=True)
    doctor = sub.add_parser("doctor"); doctor.set_defaults(func=cmd_doctor)
    for name, func in (("selfplay", cmd_selfplay), ("train", cmd_train), ("gtp", cmd_gtp), ("json-rpc", cmd_rpc)):
        p = sub.add_parser(name); p.add_argument("--config", default="configs/tiny.yaml"); p.set_defaults(func=func)
        if name == "selfplay":
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
        else: p.add_argument("--checkpoint")
    p = sub.add_parser("eval"); p.set_defaults(func=cmd_eval)
    p.add_argument("--config", default="configs/tiny.yaml")
    p.add_argument("--candidate", required=True); p.add_argument("--baseline", required=True)
    p.add_argument("--games", type=int, default=DEFAULT_GATING_GAMES)
    p.add_argument("--workers", type=int, default=1,
                   help="spawned evaluation workers; 1 keeps strict serial gatekeeper execution")
    p.add_argument("--seed", type=int, help="evaluation seed (default: config seed)")
    p.add_argument("--required-candidate-win-prop", type=float,
                   help="enable gatekeeper acceptance and early stopping at this match-point fraction; omit to play all games")
    p.add_argument("--score-utility", type=float, default=0.05,
                   help="small score prediction contribution to MCTS utility (default: 0.05)")
    p = sub.add_parser("loop"); p.set_defaults(func=cmd_loop)
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
    p.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__": main()
