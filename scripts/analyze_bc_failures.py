"""Compare saved development trajectories with human demonstration coverage, without training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from valorant_lab.bc_failure_analysis import analyze_failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scenario", default="configs/lotus_movement_combat.json")
    parser.add_argument("--report", nargs="+", required=True, help="Development report.json files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_failures(args.data, args.scenario, args.report, args.output)
    print(json.dumps({"output": args.output, "games": len(result["games"]),
                      "demonstrations": len(result["demonstrations"]),
                      "flags": result["game_flag_counts"]}, ensure_ascii=False))
