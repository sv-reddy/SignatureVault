"""
plot_arcface_margin_sensitivity.py - ArcFace Margin Sensitivity Chart Generator
===============================================================================

Reads a JSON file with margin-vs-metric values and generates:
1) PNG sensitivity chart (accuracy and EER vs ArcFace margin)
2) LaTeX table snippet for direct insertion into the paper

Expected JSON schema:
{
  "margins": [
    {"margin": 0.35, "accuracy": 0.9031, "eer": 0.0945, "precision": 0.84, "f1": 0.87},
    {"margin": 0.45, "accuracy": 0.9188, "eer": 0.0810, "precision": 0.85, "f1": 0.88},
    {"margin": 0.55, "accuracy": 0.9272, "eer": 0.0728, "precision": 0.8642, "f1": 0.8946}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
_DEFAULT_INPUT_JSON = _BACKEND_DIR / "results" / "evaluate" / "margin_sensitivity.json"
_DEFAULT_OUTPUT_DIR = _BACKEND_DIR / "results" / "evaluate"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ArcFace margin sensitivity chart and LaTeX table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-json",
        type=str,
        default=str(_DEFAULT_INPUT_JSON),
        help="Input JSON containing margin sensitivity results.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory to write chart and table.",
    )
    p.add_argument(
        "--chart-name",
        type=str,
        default="arcface_margin_sensitivity.png",
        help="Output chart filename.",
    )
    return p.parse_args()


def _render_table(rows: list[dict], table_path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\caption{ArcFace Margin Sensitivity}",
        "\\label{tab:margin_sensitivity}",
        "\\begin{center}",
        "\\begin{tabular}{|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Margin $m$} & \\textbf{Accuracy (\\%)} & \\textbf{Precision (\\%)} & \\textbf{F1-score (\\%)} & \\textbf{EER (\\%)} \\\\",
        "\\hline",
    ]

    for r in rows:
        lines.append(
            f"{r['margin']:.2f} & {r['accuracy']*100:.2f} & {r.get('precision', 0.0)*100:.2f} & {r.get('f1', 0.0)*100:.2f} & {r['eer']*100:.2f} \\\\" 
        )

    lines.extend(["\\hline", "\\end{tabular}", "\\end{center}", "\\end{table}"])
    table_path.write_text("\n".join(lines), encoding="utf-8")


def _render_chart(rows: list[dict], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for chart generation. Install with: pip install matplotlib"
        ) from exc

    rows = sorted(rows, key=lambda x: float(x["margin"]))
    margins = [float(r["margin"]) for r in rows]
    acc = [float(r["accuracy"]) * 100.0 for r in rows]
    eer = [float(r["eer"]) * 100.0 for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax1.plot(margins, acc, marker="o", linewidth=2.2, label="Accuracy", color="#1f77b4")
    ax1.set_xlabel("ArcFace margin m")
    ax1.set_ylabel("Accuracy (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(margins, eer, marker="s", linewidth=2.0, label="EER", color="#d62728")
    ax2.set_ylabel("EER (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    best_idx = max(range(len(rows)), key=lambda i: acc[i] - eer[i])
    ax1.scatter([margins[best_idx]], [acc[best_idx]], color="#2ca02c", zorder=5)
    ax1.annotate(
        f"best m={margins[best_idx]:.2f}",
        (margins[best_idx], acc[best_idx]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color="#2ca02c",
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    input_json = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_json.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {input_json}\n"
            "Create this file with margin-wise metrics first."
        )

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = payload.get("margins", [])
    if not rows:
        raise ValueError("No 'margins' entries found in input JSON.")

    rows = sorted(rows, key=lambda x: float(x["margin"]))

    chart_path = output_dir / args.chart_name
    table_path = output_dir / "arcface_margin_sensitivity_table.tex"

    _render_chart(rows, chart_path)
    _render_table(rows, table_path)

    out_payload = {
        "source": str(input_json),
        "chart": str(chart_path),
        "table": str(table_path),
        "margins": rows,
    }
    report_path = output_dir / "arcface_margin_sensitivity_report.json"
    report_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print("ArcFace margin sensitivity artifacts generated:")
    print(f"  chart : {chart_path}")
    print(f"  table : {table_path}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
