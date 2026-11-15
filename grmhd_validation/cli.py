import os
os.umask(0o007)


def main(argv=None):
    os.umask(0o007)
    import argparse
    import json
    from pathlib import Path

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

        def exit(self, status=0, message=None):
            if status == 0:
                print(json.dumps({"ok": True, "next": "gval doctor"}, sort_keys=True))
            raise SystemExit(status)

    parser = Parser(prog="gval", description="Shared GRMHD validation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only storage, split, protocol, and environment checks")
    command = sub.add_parser("checksums", help="Owner-only per-file hashing on Slurm compute nodes")
    command.add_argument("--split-file", required=True)
    command.add_argument("--data-root")
    command = sub.add_parser("evaluate", help="Evaluate an adapter in a recorded Slurm attempt")
    command.add_argument("--request", required=True)
    command.add_argument("--split-id", required=True)
    command.add_argument("--split", choices=("validation", "test"), required=True)
    command.add_argument("--tier", choices=("quick", "full"), required=True)
    command.add_argument("--device", choices=("cpu", "cuda"))
    command.add_argument("--data-root")
    command.add_argument("--no-verify-data", action="store_true")
    command = sub.add_parser("reference", help="Owner-only persistence golden on a Slurm CPU")
    command.add_argument("adapter", choices=("persistence",))
    command.add_argument("--split-id", required=True)
    command.add_argument("--split", choices=("validation",), default="validation")
    command.add_argument("--tier", choices=("quick",), default="quick")
    command.add_argument("--device", choices=("cpu",), default="cpu")
    command.add_argument("--data-root")
    command = sub.add_parser("conformance", help="Check persistence against the platform golden on Slurm CPU")
    command.add_argument("--split-id", required=True)
    command.add_argument("--data-root")
    command = sub.add_parser("verify", help="Read-only bundle verification or numerical comparison")
    command.add_argument("--compare", action="store_true")
    command.add_argument("runs", nargs="+")
    command = sub.add_parser("seal", help="Owner recovery of an unsealed COMPLETE run")
    command.add_argument("run")
    command = sub.add_parser("new-adapter", help="Write a template and request in your directory")
    command.add_argument("directory")
    try:
        args = parser.parse_args(argv)
        if args.command == "doctor":
            from .doctor import doctor
            verdict = doctor(Path(__file__).resolve().parents[1])
        elif args.command == "checksums":
            from .checksums import generate
            verdict = generate(args.split_file, args.data_root)
        elif args.command == "evaluate":
            from .runner import evaluate
            verdict = evaluate(args.request, args.split_id, args.split, args.tier, args.device, args.data_root, args.no_verify_data)
        elif args.command == "reference":
            from .conformance import reference
            verdict = reference(args.split_id, args.tier, args.split, args.device, data_root=args.data_root)
        elif args.command == "conformance":
            from .conformance import conformance
            verdict = conformance(args.split_id, data_root=args.data_root)
        elif args.command == "verify":
            from .verify import verify_run, compare_runs
            expected = 2 if args.compare else 1
            if len(args.runs) != expected:
                raise ValueError("verify requires one run, or --compare and two runs")
            verdict = compare_runs(*args.runs) if args.compare else verify_run(args.runs[0])
        elif args.command == "new-adapter":
            from .templates import new_adapter
            verdict = new_adapter(args.directory)
        elif args.command == "seal":
            from .verify import seal_run
            verdict = seal_run(args.run)
    except Exception as error:
        verdict = {"ok": False, "next": "Resolve the reported error and rerun the command; consult docs/milestone_2.md", "error": str(error)}
    print(json.dumps(verdict, sort_keys=True, allow_nan=False))
    return 0 if verdict["ok"] else 1
