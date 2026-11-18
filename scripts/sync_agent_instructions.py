"""Generate/check the tool-agnostic instructions without transforming content."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    source=(repo/'AGENTS.md').read_bytes();target=repo/'docs/agents/generic.md'
    if args.check:
        if not target.is_file() or target.read_bytes()!=source:raise SystemExit('docs/agents/generic.md differs; run scripts/sync_agent_instructions.py')
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source)


if __name__=='__main__':main()
