"""Probe or dry-run one EEG/MEG design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from react_agent.eeg_training.launch import launch_design, parse_design
from react_agent.eeg_training.protocol import data_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m react_agent.eeg_training.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--dataset", default="eeg", choices=["eeg", "meg"])
        cmd.add_argument("--exp-setting", default="intra-subject", choices=["intra-subject", "inter-subject"])
        cmd.add_argument("--subject", default="sub-01")
        cmd.add_argument("--epochs", type=int, default=50)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--out", required=True)
        if name == "run":
            cmd.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch probe or run."""
    args = _parser().parse_args(argv)
    design = parse_design(
        {
            "dataset": args.dataset,
            "exp_setting": args.exp_setting,
            "subject": args.subject,
            "epochs": str(args.epochs),
            "seed": str(args.seed),
        }
    )
    action = "probe" if args.cmd == "probe" or getattr(args, "dry_run", False) else "run"
    if args.cmd == "run" and args.dry_run:
        action = "dry_run"
    payload = launch_design(design, Path(args.out), action, data_root())
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] or action != "run" else 2


if __name__ == "__main__":
    raise SystemExit(main())
