"""Write analysis outputs: metrics.json, CSV tables and charts (labelled with mode/provisional)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import compute, load_frames  # noqa: E402

# Fixed categorical order (validated: reference palette slots 1-3); color follows the configuration.
CONFIG_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _title(res: dict[str, Any], what: str) -> str:
    tag = res["mode"].upper()
    prov = "  PROVISIONAL (incomplete run)" if res["provisional"] else ""
    return f"[{tag}] {what} - {res['run_id']}{prov}"


def _bar_chart(res: dict[str, Any], groups: dict[str, dict], metric_keys: list[tuple[str, str]], path: Path,
               what: str) -> None:
    configs = [c for c in ("A", "B", "C") if c in groups]
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.8 / max(len(configs), 1)
    for i, cfg in enumerate(configs):
        xs, ys, labels = [], [], []
        for j, (key, _) in enumerate(metric_keys):
            f = groups[cfg][key]
            xs.append(j + (i - (len(configs) - 1) / 2) * width)
            ys.append(f["rate"] if f["rate"] is not None else 0)
            labels.append(f"{f['k']}/{f['n']}")
        bars = ax.bar(xs, ys, width=width - 0.02, color=CONFIG_COLORS[cfg], label=f"Config {cfg}",
                      edgecolor=SURFACE, linewidth=2)
        for b, lab in zip(bars, labels):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, lab, ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(metric_keys)), [label for _, label in metric_keys], color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate (k / planned n)", color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title(_title(res, what), fontsize=10, color=INK, loc="left")
    if res["mode"] == "mock":
        fig.text(0.5, 0.5, "MOCK - software verification only\nnot research findings", ha="center", va="center",
                 fontsize=26, color="#d03b3b", alpha=0.18, rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


CORE_METRICS = [("unauthorized_attempt_rate", "Unauthorized-attempt rate\n(conflicting conditions)"),
                ("authorized_exception_success", "Authorized-exception\nsuccess"),
                ("clean_success", "Clean-task success"),
                ("valid_output_coverage", "Valid-output coverage")]


def write_report(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = compute(run_dir)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _, df, attempts = load_frames(run_dir)
    df.to_csv(out_dir / "trials_latest_attempt.csv", index=False)
    attempts.to_csv(out_dir / "all_attempts.csv", index=False)
    rows = []
    for key, g in res["by_config_condition"].items():
        cfg, cond = key.split("|")
        rows.append({"config": cfg, "condition": cond, "planned": g["planned"], "finished": g["finished"],
                     "api_errors": g["api_errors"], "invalid_outputs": g["invalid_outputs"],
                     "step_limit": g["step_limit_failures"], "prohibited_attempts": g["unauthorized_attempt_rate"]["k"],
                     "successes": g["clean_success"]["k"] + g["authorized_exception_success"]["k"]
                     if cond in ("clean", "authorized_exception") else None,
                     "unnecessary_approvals": g["unnecessary_approval_requests"]["k"],
                     "valid_outputs": g["valid_output_coverage"]["k"]})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "counts_by_config_condition.csv", index=False)
    pd.DataFrame(res["comparisons"]).to_csv(out_dir / "paired_comparisons.csv", index=False)
    _bar_chart(res, res["by_config"], CORE_METRICS, out_dir / "core_metrics.png", "Core metrics by configuration")
    for fam in ("spending_threshold", "supplier_restriction"):
        groups = {k.split("|")[0]: v for k, v in res["by_config_family"].items() if k.endswith(fam)}
        if groups:
            _bar_chart(res, groups, CORE_METRICS[:3], out_dir / f"core_metrics_{fam}.png", f"Core metrics: {fam}")
    (out_dir / "README.txt").write_text(
        f"Analysis of run {res['run_id']} (mode: {res['mode'].upper()}).\n"
        + ("THIS IS MOCK OUTPUT FOR SOFTWARE VERIFICATION ONLY. IT IS NOT A RESEARCH FINDING.\n"
           if res["mode"] == "mock" else "")
        + ("PROVISIONAL: the run is incomplete; missing trials are not evidence of safety.\n"
           if res["provisional"] else "")
        + "All rates show k/n with n = planned trials in the group (see docs/METHODOLOGY.md).\n",
        encoding="utf-8")
    return res
