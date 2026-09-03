"""CLI: python -m aerosentry.cli "Schedule a thermal inspection for D-01" [--approve|--reject]"""
import argparse
import json

from .agents import Orchestrator


def main():
    p = argparse.ArgumentParser(description="AeroSentry agentic drone-ops CLI")
    p.add_argument("message")
    p.add_argument("--approve", action="store_true", help="auto-approve any high-risk action the agent proposes")
    p.add_argument("--reject", action="store_true", help="auto-reject any high-risk action")
    a = p.parse_args()
    orch = Orchestrator()
    out = orch.run(a.message)
    if out["pending_approval"] and (a.approve or a.reject):
        out = orch.approve(out["thread_id"], a.approve)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
