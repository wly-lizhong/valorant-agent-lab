"""Diagnose a saved BC policy or run an explicitly isolated train-only fit check."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--tiny-fit", action="store_true")
    parser.add_argument("--data", required=True)
    parser.add_argument("--scenario", default="configs/lotus_movement_combat.json")
    parser.add_argument("--output", required=True, help="Fresh JSON file for a model; fresh directory for --tiny-fit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Default: 128 for saved models, 32 for tiny-fit")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Per split: 0 means all. Tiny-fit: 0 means 128, maximum 128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-test-data", action="store_true", help="Synthetic pipeline fixtures only")
    parser.add_argument("--policy-variant", choices=["normalized", "normalized-relu"], default="normalized")
    parser.add_argument("--class-balance", choices=["none", "sqrt"], default="sqrt")
    parser.add_argument("--class-weight-cap", type=float, default=8.)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=300, help="Tiny-fit optimizer updates, maximum 300")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 32 if args.tiny_fit else 128
    from valorant_lab.bc_diagnostics import diagnose_model, tiny_fit
    report = tiny_fit(args) if args.tiny_fit else diagnose_model(args)
    if args.tiny_fit:
        def brief(value):
            return {name: value["metrics"][name] for name in
                    ("joint_accuracy", "unique_joint_actions", "fire", "turn", "stop")}
        summary = {"mode": "tiny_fit_diagnostic_only", "samples": report["sample_count"],
                   "before": brief(report["before"]), "after": brief(report["after"]),
                   "shuffled_minus_aligned_nll": report["after"]["visual_dependency"]["shuffled_minus_aligned_nll"]}
    else:
        summary = {split: {"samples": value["samples"],
                           "unique_joint_actions": value["metrics"]["unique_joint_actions"],
                           "fire": value["metrics"]["fire"], "turn": value["metrics"]["turn"],
                           "stop": value["metrics"]["stop"],
                           "shuffled_minus_aligned_nll": value["visual_dependency"]["shuffled_minus_aligned_nll"]}
                   for split, value in report["splits"].items()}
    print(json.dumps({"output": args.output, "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
