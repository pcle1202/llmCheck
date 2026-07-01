from __future__ import annotations

import dataclasses
import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.scorer import EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _fmt_f(v: float | None, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "N/A"


def _escape(text: str | None) -> str:
    if text is None:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _model_stats(results: list[EvalResult]) -> dict[str, dict]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {"similarities": [], "latencies": [], "safe_vals": []}
    )
    for r in results:
        b = buckets[r.model_name]
        if r.similarity_score is not None:
            b["similarities"].append(r.similarity_score)
        if r.latency is not None:
            b["latencies"].append(r.latency)
        if r.safe is not None:
            b["safe_vals"].append(r.safe)

    stats: dict[str, dict] = {}
    for model, b in buckets.items():
        safe_vals = b["safe_vals"]
        stats[model] = {
            "avg_similarity": _avg(b["similarities"]),
            "avg_latency": _avg(b["latencies"]),
            "pct_safe": (sum(safe_vals) / len(safe_vals)) if safe_vals else None,
        }
    return stats


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
    body  { font-family: Arial, sans-serif; margin: 2rem; color: #333; }
    h1    { color: #1a1a2e; }
    h2    { color: #16213e; margin-top: 2.5rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th    { background: #1a1a2e; color: #fff; padding: 8px 12px; text-align: left; }
    td    { padding: 8px 12px; border-bottom: 1px solid #ddd; vertical-align: top; }
    tr:hover td { background: #f0f4ff; }
    .safe-true  td { background: #d4edda; }
    .safe-false td { background: #f8d7da; }
    .safe-true  td:hover,
    .safe-false td:hover { filter: brightness(0.95); }
    .tag-safe   { color: #155724; font-weight: bold; }
    .tag-unsafe { color: #721c24; font-weight: bold; }
    .na         { color: #999; font-style: italic; }
"""


def _summary_rows(stats: dict[str, dict]) -> str:
    rows = ""
    for model, s in stats.items():
        rows += (
            f"<tr>"
            f"<td>{_escape(model)}</td>"
            f"<td>{_fmt_f(s['avg_similarity'])}</td>"
            f"<td>{_fmt_f(s['avg_latency'])} s</td>"
            f"<td>{_fmt_pct(s['pct_safe'])}</td>"
            f"</tr>\n"
        )
    return rows


def _factual_rows(results: list[EvalResult]) -> str:
    rows = ""
    for r in (r for r in results if r.safe is None):
        rows += (
            f"<tr>"
            f"<td>{_escape(r.prompt)}</td>"
            f"<td>{_escape(r.model_name)}</td>"
            f"<td>{_escape(r.expected)}</td>"
            f"<td>{_escape(r.actual)}</td>"
            f"<td>{'Yes' if r.exact_match else 'No'}</td>"
            f"<td>{_fmt_f(r.similarity_score)}</td>"
            f"</tr>\n"
        )
    return rows


def _adversarial_rows(results: list[EvalResult]) -> str:
    rows = ""
    for r in (r for r in results if r.safe is not None):
        css_class = "safe-true" if r.safe else "safe-false"
        tag = (
            '<span class="tag-safe">SAFE</span>'
            if r.safe
            else '<span class="tag-unsafe">UNSAFE</span>'
        )
        rows += (
            f'<tr class="{css_class}">'
            f"<td>{_escape(r.prompt)}</td>"
            f"<td>{_escape(r.model_name)}</td>"
            f"<td>{_escape(r.actual)}</td>"
            f"<td>{tag}</td>"
            f"<td>{_escape(r.unsafe_reason)}</td>"
            f"<td>{_escape(r.judge_response)}</td>"
            f"</tr>\n"
        )
    return rows


def _build_html(results: list[EvalResult], stats: dict[str, dict]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LLM Evaluation Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>LLM Evaluation Report</h1>

  <h2>Model Summary</h2>
  <table>
    <thead><tr>
      <th>Model</th>
      <th>Avg Similarity (factual)</th>
      <th>Avg Latency</th>
      <th>Adversarial Safety Rate</th>
    </tr></thead>
    <tbody>
{_summary_rows(stats)}    </tbody>
  </table>

  <h2>Factual / Open-Ended Cases</h2>
  <table>
    <thead><tr>
      <th>Prompt</th><th>Model</th><th>Expected</th>
      <th>Actual</th><th>Exact Match</th><th>Similarity</th>
    </tr></thead>
    <tbody>
{_factual_rows(results)}    </tbody>
  </table>

  <h2>Adversarial Cases</h2>
  <table>
    <thead><tr>
      <th>Prompt</th><th>Model</th><th>Actual Response</th><th>Safe</th><th>Unsafe Reason</th><th>Judge Response</th>
    </tr></thead>
    <tbody>
{_adversarial_rows(results)}    </tbody>
  </table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Terminal summary (stdlib only)
# ---------------------------------------------------------------------------

def _print_terminal_summary(stats: dict[str, dict]) -> None:
    col_widths = [24, 18, 14, 16]
    headers = ["Model", "Avg Similarity", "Avg Latency", "Safety Rate"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def fmt_row(*cells: str) -> str:
        parts = (f" {str(c):<{w}} " for c, w in zip(cells, col_widths))
        return "|" + "|".join(parts) + "|"

    print()
    print(sep)
    print(fmt_row(*headers))
    print(sep)
    for model, s in stats.items():
        print(fmt_row(
            model[:col_widths[0]],
            _fmt_f(s["avg_similarity"]),
            f"{_fmt_f(s['avg_latency'])} s",
            _fmt_pct(s["pct_safe"]),
        ))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    results: list[EvalResult],
    output_dir: str = "reports",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2)

    stats = _model_stats(results)

    html_path = os.path.join(output_dir, "results.html")
    with open(html_path, "w") as f:
        f.write(_build_html(results, stats))

    _print_terminal_summary(stats)
    print(f"Report written to {output_dir}/")
