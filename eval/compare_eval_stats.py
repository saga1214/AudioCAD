import argparse
import json
import os


def load_stats(result_dir, judge_model):
    stats_path = os.path.join(result_dir, f"{judge_model}_stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_axis_delta_report(baseline_stats, cad_stats):
    axes = sorted(set(baseline_stats.get("by_axis", {}).keys()) | set(cad_stats.get("by_axis", {}).keys()))
    report = {}
    for axis in axes:
        baseline_axis = baseline_stats.get("by_axis", {}).get(axis, {})
        cad_axis = cad_stats.get("by_axis", {}).get(axis, {})
        report[axis] = {
            "baseline_score": baseline_axis.get("score", 0.0),
            "cad_score": cad_axis.get("score", 0.0),
            "delta_score": cad_axis.get("score", 0.0) - baseline_axis.get("score", 0.0),
            "baseline_passed": baseline_axis.get("passed", 0),
            "baseline_total": baseline_axis.get("total", 0),
            "cad_passed": cad_axis.get("passed", 0),
            "cad_total": cad_axis.get("total", 0),
        }
    return report


def build_inference_memory_breakdown_report(baseline_stats, cad_stats):
    baseline_breakdown = baseline_stats.get("inference_memory_breakdown") or {}
    cad_breakdown = cad_stats.get("inference_memory_breakdown") or {}
    subset_names = sorted(set(baseline_breakdown.keys()) | set(cad_breakdown.keys()))
    if not subset_names:
        return None

    report = {}
    for subset_name in subset_names:
        baseline_subset = baseline_breakdown.get(subset_name, {})
        cad_subset = cad_breakdown.get(subset_name, {})
        report[subset_name] = {
            "baseline_score": baseline_subset.get("score", 0.0),
            "cad_score": cad_subset.get("score", 0.0),
            "delta_score": cad_subset.get("score", 0.0) - baseline_subset.get("score", 0.0),
            "baseline_passed": baseline_subset.get("passed", 0),
            "baseline_total": baseline_subset.get("total", 0),
            "cad_passed": cad_subset.get("passed", 0),
            "cad_total": cad_subset.get("total", 0),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--cad_dir", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-5-nano-2025-08-07")
    args = parser.parse_args()

    baseline_stats = load_stats(args.baseline_dir, args.judge_model)
    cad_stats = load_stats(args.cad_dir, args.judge_model)

    report = {
        "baseline_dir": args.baseline_dir,
        "cad_dir": args.cad_dir,
        "judge_model": args.judge_model,
        "baseline_overall_score": baseline_stats.get("overall", {}).get("score", 0.0),
        "cad_overall_score": cad_stats.get("overall", {}).get("score", 0.0),
        "overall_delta_score": cad_stats.get("overall", {}).get("score", 0.0)
        - baseline_stats.get("overall", {}).get("score", 0.0),
        "axis_report": build_axis_delta_report(baseline_stats, cad_stats),
        "inference_memory_breakdown_report": build_inference_memory_breakdown_report(
            baseline_stats, cad_stats
        ),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
