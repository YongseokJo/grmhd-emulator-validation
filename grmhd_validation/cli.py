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
    command.add_argument("--predictions")
    command = sub.add_parser("reference", help="Owner-only persistence golden on a Slurm CPU")
    command.add_argument("adapter", choices=("persistence", "channel-scales", "truth"))
    command.add_argument("--split-id", required=True)
    command.add_argument("--split", choices=("validation","test"), default="validation")
    command.add_argument("--tier", choices=("quick","full"), default="quick")
    command.add_argument("--device", choices=("cpu",), default="cpu")
    command.add_argument("--data-root")
    command = sub.add_parser("conformance", help="Check persistence against the platform golden on Slurm CPU")
    command.add_argument("--split-id", required=True)
    command.add_argument("--data-root")
    command.add_argument('--tier',choices=('quick','full','all'))
    command = sub.add_parser("verify", help="Read-only bundle verification or numerical comparison")
    command.add_argument("--compare", action="store_true")
    command.add_argument("runs", nargs="+")
    command = sub.add_parser("seal", help="Owner recovery of an unsealed COMPLETE run")
    command.add_argument("run")
    command = sub.add_parser("new-adapter", help="Write a template and request in your directory")
    command.add_argument("directory")
    command = sub.add_parser('task', help='Host: freeze an agent prediction task outside shared storage')
    for flag in ('request','split-id','prompt-id','output'):command.add_argument('--'+flag, required=True)
    command.add_argument('--split',choices=('validation','test'),required=True)
    command.add_argument('--protocol',choices=('A','B'),required=True)
    command.add_argument('--data-root')
    command = sub.add_parser('packet',help='Host: create, advance or collect isolated prediction packets')
    packet_sub=command.add_subparsers(dest='packet_command',required=True)
    create=packet_sub.add_parser('create');create.add_argument('--task',required=True)
    create.add_argument('--kind',choices=('rollout','single_step'),required=True)
    create.add_argument('--start',type=int,required=True);create.add_argument('--output',required=True)
    advance=packet_sub.add_parser('advance');advance.add_argument('--task',required=True);advance.add_argument('--packet',required=True)
    collect=packet_sub.add_parser('collect');collect.add_argument('--task',required=True);collect.add_argument('--output',required=True)
    command = sub.add_parser('record-training',help='Record declared checkpoint training provenance')
    for flag in ('request','split-id','config'):command.add_argument('--'+flag,required=True)
    command.add_argument('--log')
    command = sub.add_parser('check-adapter',help='Validate physical adapter output and deterministic eval calls on Slurm')
    command.add_argument('adapter');command.add_argument('--checkpoint')
    command.add_argument('--split-id',required=True);command.add_argument('--options',default='{}')
    command.add_argument('--device',choices=('cpu','cuda'),default='cpu');command.add_argument('--data-root')
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
            verdict = evaluate(args.request, args.split_id, args.split, args.tier, args.device, args.data_root, args.no_verify_data, predictions=args.predictions)
        elif args.command == 'task':
            from .predictions import task_create
            verdict=task_create(args.request,args.split_id,args.split,args.protocol,args.prompt_id,args.output,args.data_root)
        elif args.command == 'packet':
            from .predictions import packet_create,packet_advance,packet_collect
            if args.packet_command=='create':verdict=packet_create(args.task,args.kind,args.start,args.output)
            elif args.packet_command=='advance':verdict=packet_advance(args.task,args.packet)
            else:verdict=packet_collect(args.task,args.output)
        elif args.command == 'record-training':
            from .training import record_training
            verdict=record_training(args.request,args.split_id,args.config,args.log)
        elif args.command == 'check-adapter':
            from .check_adapter import check_adapter
            verdict=check_adapter(args.adapter,args.checkpoint,args.split_id,json.loads(args.options),args.device,args.data_root)
        elif args.command == "reference":
            if args.adapter=='channel-scales':
                from .channel_scales import generate
                verdict=generate(args.split_id,data_root=args.data_root)
            else:
                from .conformance import reference
                verdict = reference(args.split_id, args.tier, args.split, args.device, data_root=args.data_root,truth_only=args.adapter=='truth')
        elif args.command == "conformance":
            from .conformance import conformance
            from .protocol import read_protocol,full_tier_ready
            protocol,_=read_protocol(Path(__file__).resolve().parents[1]/'protocols/forecast_v1.json')
            tier=args.tier or ('all' if full_tier_ready(protocol) else 'quick')
            results=[conformance(args.split_id,data_root=args.data_root,tier=value) for value in (['quick','full'] if tier=='all' else [tier])]
            verdict=results[0] if len(results)==1 else {'ok':all(r['ok'] for r in results),'next':'gval new-adapter <dir>','tiers':results}
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
