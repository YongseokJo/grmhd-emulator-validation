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

    parser = Parser(prog="gval", description="Shared GRMHD validation pipeline (milestone 1)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only storage, split, protocol, and environment checks")
    command = sub.add_parser("checksums", help="Owner-only per-file hashing on Slurm compute nodes")
    command.add_argument("--split-file", required=True)
    command.add_argument("--data-root")
    try:
        args = parser.parse_args(argv)
        if args.command == "doctor":
            from .doctor import doctor
            verdict = doctor(Path(__file__).resolve().parents[1])
        else:
            from .checksums import generate
            verdict = generate(args.split_file, args.data_root)
    except Exception as error:
        verdict = {"ok": False, "next": "Resolve the reported error and rerun the command; consult docs/milestone_1.md", "error": str(error)}
    print(json.dumps(verdict, sort_keys=True, allow_nan=False))
    return 0 if verdict["ok"] else 1
