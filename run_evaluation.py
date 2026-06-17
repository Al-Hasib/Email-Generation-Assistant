"""
Main entry point: Runs evaluation for all models (A, B, C),
generates reports, and outputs comparison analysis.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from models.model_a import ModelA
from models.model_b import ModelB
from models.model_c import ModelC
from evaluation.evaluator import run_evaluation
from evaluation.report import generate_report, generate_html_report


def main():
    config = load_config("config.yaml" if os.path.exists("config.yaml") else None)

    if not config.groq_api_key:
        print("ERROR: GROQ_API_KEY not set. Create a .env file (see .env.example).")
        sys.exit(1)

    output_dir = config.results_dir
    os.makedirs(output_dir, exist_ok=True)

    all_reports = {}

    models_to_run = [
        ("Model A (COT + Review)", ModelA(config.model_a)),
        ("Model B (Role-Playing)", ModelB(config.model_b)),
        ("Model C (Self-Reflection)", ModelC(config.model_c)),
    ]

    for label, model in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running {label}...")
        print(f"{'='*60}")
        results = run_evaluation(model, config)
        report = generate_report(results, output_dir, label.lower().replace(" ", "_").replace("(", "").replace(")", ""))
        all_reports[label] = report
        s = report["average_scores"]
        print(f"  -> FRA: {s['fact_recall_accuracy']:.3f} | TAS: {s['tone_adherence_score']:.3f} | OQS: {s['overall_quality_score']:.3f} | Overall: {s['overall_average']:.3f}")

    # --- HTML report ---
    html_path = generate_html_report(all_reports, output_dir)
    print(f"\nHTML report: {html_path}")

    # --- Comparison analysis ---
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    analysis = _generate_comparison(all_reports)
    print(analysis)

    analysis_path = os.path.join(output_dir, "comparison_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"\nAnalysis written to {analysis_path}")

    comparison_json_path = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"Full results written to {comparison_json_path}")


def _generate_comparison(all_reports: dict) -> str:
    names = list(all_reports.keys())
    lines = []
    lines.append("# Model Comparison Analysis\n\n")

    for name in names:
        r = all_reports[name]
        lines.append(f"- **{name}:** {r['model_name']} ({r['strategy_name']})\n")

    lines.append("\n## Average Scores\n\n")
    header = "| Metric |" + "|".join(f" {n} " for n in names) + "|"
    sep = "|--------" + "|--------" * len(names) + "|"
    lines.append(header + "\n" + sep + "\n")

    metrics = [
        ("Fact Recall Accuracy (FRA)", "fact_recall_accuracy"),
        ("Tone Adherence Score (TAS)", "tone_adherence_score"),
        ("Overall Quality Score (OQS)", "overall_quality_score"),
    ]

    scores = {}
    for label, key in metrics:
        vals = [all_reports[n]["average_scores"][key] for n in names]
        scores[key] = vals
        row = f"| {label} " + "".join(f"| {v:.4f} " for v in vals) + "|"
        lines.append(row + "\n")

    overalls = [all_reports[n]["average_scores"]["overall_average"] for n in names]
    row = "| **Overall Average** " + "".join(f"| **{v:.4f}** " for v in overalls) + "|"
    lines.append(row + "\n")

    best_idx = overalls.index(max(overalls))
    worst_idx = overalls.index(min(overalls))

    lines.append(f"\n## Which Model Performed Best?\n\n")
    lines.append(f"**{names[best_idx]}** achieved the highest overall average ({overalls[best_idx]:.4f}).\n")

    lines.append(f"\n## Biggest Failure Mode of the Lowest-Ranking Model\n\n")
    lines.append(f"**{names[worst_idx]}** had the weakest overall score ({overalls[worst_idx]:.4f}).\n")

    worst_report = all_reports[names[worst_idx]]
    worst_metric = min(
        [
            ("Fact Recall", worst_report["average_scores"]["fact_recall_accuracy"]),
            ("Tone Adherence", worst_report["average_scores"]["tone_adherence_score"]),
            ("Overall Quality", worst_report["average_scores"]["overall_quality_score"]),
        ],
        key=lambda x: x[1],
    )
    lines.append(
        f"Its weakest area was **{worst_metric[0]}** (score: {worst_metric[1]:.4f}). "
    )
    if worst_metric[0] == "Fact Recall":
        lines.append(
            "This indicates the model frequently omitted or misrepresented "
            "key facts, which is critical for an email assistant."
        )
    elif worst_metric[0] == "Tone Adherence":
        lines.append(
            "The model struggled to calibrate language, formality, and style "
            "to the requested tone."
        )
    else:
        lines.append(
            "There were issues with structure, clarity, grammar, or call-to-action "
            "effectiveness."
        )

    lines.append(f"\n\n## Recommendation for Production\n\n")
    lines.append(
        f"I recommend **{names[best_idx]}** for production. "
    )
    if "Self-Reflection" in names[best_idx] or "Review Loop" in names[best_idx]:
        lines.append(
            "The iterative self-reflection approach (Plan → Write → Review → Rewrite) "
            "provides built-in quality control that catches and fixes issues before "
            "the email is finalised. While it uses more tokens per generation, the "
            "improvement in fact recall and overall quality makes it worth the cost "
            "for a production system where accuracy matters."
        )
    elif "Role-Playing" in names[best_idx]:
        lines.append(
            "The role-playing approach matches or exceeds more complex strategies "
            "while being simpler to maintain. It is the best choice if API cost "
            "or latency are primary concerns."
        )
    else:
        lines.append(
            "This strategy provides the best balance of quality, cost, and reliability."
        )

    return "".join(lines)


if __name__ == "__main__":
    main()
