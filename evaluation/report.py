"""
Report Generator: Produces CSV, JSON, and HTML evaluation reports.
"""

import csv
import json
import os
from typing import List, Dict

from evaluation.evaluator import ScenarioResult


def _bar(value: float, max_val: float = 1.0, color: str = "#4CAF50", width: int = 20) -> str:
    filled = round((value / max_val) * width) if max_val else 0
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="width:{width * 10}px;height:18px;background:#eee;border-radius:3px;overflow:hidden;">'
        f'<div style="width:{filled * 10}px;height:18px;background:{color};border-radius:3px;"></div>'
        f'</div>'
        f'<span style="font-size:13px;color:#555;">{value:.3f}</span>'
        f'</div>'
    )


def _color_for(value: float) -> str:
    if value >= 0.85:
        return "#4CAF50"
    if value >= 0.65:
        return "#FF9800"
    return "#f44336"


def generate_html_report(
    all_reports: Dict[str, dict], output_dir: str
) -> str:
    rows_html = ""
    model_names = list(all_reports.keys())
    metrics = ["fact_recall_accuracy", "tone_adherence_score", "overall_quality_score"]
    metric_labels = ["Fact Recall (FRA)", "Tone Adherence (TAS)", "Overall Quality (OQS)"]

    for i, scenario in enumerate(
        all_reports[model_names[0]]["results"]
    ):
        sid = scenario["scenario_id"]
        intent = scenario["intent"]
        tone = scenario["tone"]
        cells = f"<td>{sid}</td><td>{intent}</td><td>{tone}</td>"
        for mn in model_names:
            r = all_reports[mn]["results"][i]
            avg = (
                r["fact_recall_accuracy"]
                + r["tone_adherence_score"]
                + r["overall_quality_score"]
            ) / 3
            cells += (
                f'<td>{_bar(r["fact_recall_accuracy"], color=_color_for(r["fact_recall_accuracy"]))}</td>'
                f'<td>{_bar(r["tone_adherence_score"], color=_color_for(r["tone_adherence_score"]))}</td>'
                f'<td>{_bar(r["overall_quality_score"], color=_color_for(r["overall_quality_score"]))}</td>'
                f'<td><strong>{avg:.3f}</strong></td>'
            )
        rows_html += f"<tr>{cells}</tr>\n"

    # Summary table
    summary_rows = ""
    for label, key in zip(metric_labels, metrics):
        cells = f"<td><strong>{label}</strong></td>"
        for mn in model_names:
            s = all_reports[mn]["average_scores"][key]
            cells += f'<td>{_bar(s, color=_color_for(s))}</td>'
        summary_rows += f"<tr>{cells}</tr>\n"

    overall_cells = "<td><strong>Overall Average</strong></td>"
    for mn in model_names:
        s = all_reports[mn]["average_scores"]["overall_average"]
        overall_cells += f'<td>{_bar(s, color=_color_for(s))}</td>'
    summary_rows += f"<tr>{overall_cells}</tr>\n"

    # Strategy names header
    strat_cells = "".join(
        f'<th style="font-weight:400;font-size:12px;color:#666;" colspan="4">{all_reports[mn]["strategy_name"]}</th>'
        for mn in model_names
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Email Generation Assistant — Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px; background: #f8f9fa; color: #333; }}
h1, h2 {{ color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }}
th {{ background: #1a1a2e; color: #fff; font-weight: 600; }}
tr:hover {{ background: #f1f3f5; }}
.meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
.section {{ margin: 28px 0 12px; }}
</style>
</head>
<body>
<h1>Email Generation Assistant — Evaluation Report</h1>
<div class="meta">
  <p>Models evaluated: {', '.join(model_names)}</p>
  <p>Scenarios: {len(all_reports[model_names[0]]["results"])}</p>
</div>

<h2>Average Scores</h2>
<table>
<tr><th style="width:200px;">Metric</th>{strat_cells}</tr>
{summary_rows}
</table>

<h2>Per-Scenario Breakdown</h2>
<table>
<tr>
  <th>#</th><th>Intent</th><th>Tone</th>
  {''.join(f'<th colspan="4" style="text-align:center;">{mn}</th>' for mn in model_names)}
</tr>
<tr style="font-size:12px;color:#666;">
  <th colspan="3"></th>
  {''.join('<th>FRA</th><th>TAS</th><th>OQS</th><th>Avg</th>' for _ in model_names)}
</tr>
{rows_html}
</table>

<h2>Metric Definitions</h2>
<ul>
  <li><strong>Fact Recall Accuracy (FRA):</strong> Keyword-match based — fraction of required key facts present in the output (≥50% keyword overlap threshold). Automated.</li>
  <li><strong>Tone Adherence Score (TAS):</strong> LLM-as-a-Judge — rates how well tone matches the request (1–5, normalized to 0–1).</li>
  <li><strong>Overall Quality Score (OQS):</strong> LLM-as-a-Judge — holistic quality: structure, clarity, grammar, CTA, fact integration (1–5, normalized to 0–1).</li>
</ul>

<h2>Prompt Strategies</h2>
<ul>
  {''.join(f'<li><strong>{mn}:</strong> {all_reports[mn]["strategy_name"]}</li>' for mn in model_names)}
</ul>
</body>
</html>"""

    path = os.path.join(output_dir, "evaluation_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def generate_report(
    results: List[ScenarioResult], output_dir: str, model_label: str
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    total_fra = 0.0
    total_tas = 0.0
    total_oqs = 0.0
    n = len(results)

    for r in results:
        fra = round(r.metrics.fact_recall_score, 4)
        tas = round(r.metrics.tone_adherence_score, 4)
        oqs = round(r.metrics.overall_quality_score, 4)
        total_fra += fra
        total_tas += tas
        total_oqs += oqs

        rows.append({
            "scenario_id": r.scenario_id,
            "intent": r.intent,
            "tone": r.tone,
            "fact_recall_accuracy": fra,
            "fact_recall_details": r.metrics.fact_recall_details,
            "tone_adherence_score": tas,
            "overall_quality_score": oqs,
            "generated_email": r.email_result.parsed_email,
        })

    avg_fra = round(total_fra / n, 4) if n else 0
    avg_tas = round(total_tas / n, 4) if n else 0
    avg_oqs = round(total_oqs / n, 4) if n else 0
    avg_overall = round((avg_fra + avg_tas + avg_oqs) / 3, 4)

    report = {
        "model_name": results[0].email_result.model_name if results else "",
        "strategy_name": results[0].email_result.strategy_name if results else "",
        "metric_definitions": {
            "fact_recall_accuracy": {
                "name": "Fact Recall Accuracy (FRA)",
                "description": "Keyword-match based — fraction of required key facts present in the output (>=50% keyword overlap threshold). Automated.",
                "range": "0.0 to 1.0",
                "higher_is_better": True,
            },
            "tone_adherence_score": {
                "name": "Tone Adherence Score (TAS)",
                "description": "LLM-as-a-Judge — rates how well the email's language, formality, and style match the requested tone on a scale of 1-5, normalized to 0-1.",
                "range": "0.0 to 1.0",
                "higher_is_better": True,
            },
            "overall_quality_score": {
                "name": "Overall Quality Score (OQS)",
                "description": "LLM-as-a-Judge — holistic quality assessment including structure, clarity, grammar, professionalism, call-to-action effectiveness, and fact integration on a scale of 1-5, normalized to 0-1.",
                "range": "0.0 to 1.0",
                "higher_is_better": True,
            },
        },
        "average_scores": {
            "fact_recall_accuracy": avg_fra,
            "tone_adherence_score": avg_tas,
            "overall_quality_score": avg_oqs,
            "overall_average": avg_overall,
        },
        "results": rows,
    }

    csv_path = os.path.join(output_dir, f"evaluation_report_{model_label}.csv")
    json_path = os.path.join(output_dir, f"evaluation_report_{model_label}.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario_id", "intent", "tone",
            "fact_recall_accuracy", "tone_adherence_score", "overall_quality_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
