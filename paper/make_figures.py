"""Generate the manuscript's data-backed figures.

Run from the repository root:

    python paper/make_figures.py

The script reads only released per-user CSV artifacts. In the working
repository it writes into the Overleaf tree; in the compact artifact it uses
``paper/figures``.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "jlens_results"
OVERLEAF_OUT = ROOT / "overleaf_wsdm2027" / "figures"
OUT = OVERLEAF_OUT if OVERLEAF_OUT.parent.exists() else ROOT / "paper" / "figures"

DATASET_COLORS = {
    "ML1M": "#0072B2",
    "Yahoo": "#D55E00",
    "Pinterest": "#009E73",
}
def _set_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _clean_gap(frame: pd.DataFrame) -> pd.Series:
    """Return the bounded behavior--mechanism gap from legacy result files."""
    if "clean_gap" in frame:
        return frame["clean_gap"]
    return np.clip(1.0 - frame["mr"], 0.0, 1.0)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval, returned in percentage points."""
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return 100.0 * (center - radius), 100.0 * (center + radius)


def _joint_tail_count(frame: pd.DataFrame) -> tuple[int, int]:
    gap = _clean_gap(frame)
    success = (frame["delta_beh"] > 0.2) & (gap > 0.5)
    return int(success.sum()), len(frame)


def _rate_label(rate: float) -> str:
    return f"{rate:.1f}".rstrip("0").rstrip(".")


def headline_results() -> Path:
    """Lead with the SASRec result and its checkpoint replication."""
    _set_style()
    fig, (matched_ax, replicate_ax) = plt.subplots(
        1, 2, figsize=(6.95, 2.15),
        gridspec_kw={"width_ratios": [0.86, 1.14]},
        constrained_layout=True,
    )

    matched = pd.read_csv(RESULTS / "ml1m_sasrec_matched_k5.csv")
    matched = matched.loc[matched["checkpoint"] == "published"].set_index(
        "method"
    )
    method_rows = [
        ("White-box search", "attack_lam1", "#D55E00"),
        ("LIME", "lime", "#009E73"),
        ("Occlusion", "occlusion", "#0072B2"),
        ("Random", "random", "#777777"),
    ]
    y_positions = np.arange(len(method_rows))[::-1]
    matched_ax.axvline(
        50, color="#b8bdc5", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0
    )
    for y, (label, method, color) in zip(y_positions, method_rows):
        value = 100.0 * float(matched.loc[method, "matched_gap"])
        matched_ax.hlines(y, 0, value, color=color, linewidth=1.4, alpha=0.7)
        matched_ax.scatter(
            value, y, s=27, color=color, edgecolor="white", linewidth=0.45,
            zorder=3,
        )
        matched_ax.text(
            value + 1.7, y, f"{value:.0f}", va="center", ha="left",
            fontsize=6.4, color=color, fontweight="bold",
        )
    matched_ax.text(
        50, 3.37, r"$G=50\%$", ha="center", va="bottom", fontsize=5.8,
        color="#777777",
    )
    matched_ax.set_yticks(y_positions, [row[0] for row in method_rows])
    matched_ax.set_xlim(0, 78)
    matched_ax.set_xticks([0, 20, 40, 60])
    matched_ax.set_ylim(-0.55, 3.55)
    matched_ax.set_xlabel(r"effect-matched gap $G$ (%)")
    matched_ax.set_title(
        "(a) Gap after matching score effects", loc="left", fontweight="bold"
    )
    matched_ax.grid(axis="x", color="#e0e2e6", linewidth=0.5, zorder=0)

    x_rows = [
        ("ML1M\nOcclusion", "ml1m", "occlusion_strong", "#0072B2", "o"),
        ("ML1M\nLIME", "ml1m", "lime_strong", "#009E73", "s"),
        ("Sports\nOcclusion", "sports", "occlusion_strong", "#0072B2", "o"),
        ("Sports\nLIME", "sports", "lime_strong", "#009E73", "s"),
    ]
    for x, (label, dataset, column, color, marker) in enumerate(x_rows):
        frame = pd.read_csv(RESULTS / f"{dataset}_sasrec_seed_strong.csv")
        values = 100.0 * frame[column].dropna().to_numpy()
        offsets = np.linspace(-0.10, 0.10, len(values))
        replicate_ax.scatter(
            x + offsets, values, s=14, marker=marker, color=color, alpha=0.45,
            linewidth=0, zorder=2,
        )
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        replicate_ax.errorbar(
            x, mean, yerr=sd, fmt=marker, markersize=5.3,
            markeredgecolor="white", markeredgewidth=0.5, color=color,
            ecolor=color, elinewidth=1.0, capsize=2.1, zorder=3,
        )
        replicate_ax.text(
            x, mean + sd + 2.0, f"{mean:.0f}", ha="center", va="bottom",
            fontsize=6.1, color=color, fontweight="bold",
        )
    replicate_ax.axvline(1.5, color="#d4d7dc", linewidth=0.8)
    replicate_ax.set_xticks(range(len(x_rows)), [row[0] for row in x_rows])
    replicate_ax.set_xlim(-0.45, 3.45)
    replicate_ax.set_ylim(0, 43)
    replicate_ax.set_yticks([0, 10, 20, 30, 40])
    replicate_ax.set_ylabel(r"events with $\Delta>0.2$, $G>0.5$ (%)")
    replicate_ax.set_title(
        "(b) Replication across checkpoints", loc="left", fontweight="bold"
    )
    replicate_ax.grid(axis="y", color="#e0e2e6", linewidth=0.5, zorder=0)
    replicate_ax.text(
        0.5, 1.0, "$n=4$ checkpoints", ha="center", va="bottom",
        fontsize=5.8, color="#777777",
    )
    replicate_ax.text(
        2.5, 1.0, "$n=3$ checkpoints", ha="center", va="bottom",
        fontsize=5.8, color="#777777",
    )

    for ax in (matched_ax, replicate_ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "headline_results.pdf"
    fig.savefig(target, bbox_inches="tight")
    # GitHub does not render PDF figures inline, so keep a raster copy of the
    # headline result for the repository landing page.
    fig.savefig(
        OUT / "headline_results.png", dpi=220, bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return target


def results_overview() -> Path:
    """Summarize cohort-level large-gap rates for static and sequential models."""
    _set_style()
    fig, (static_ax, sasrec_ax) = plt.subplots(
        1, 2, figsize=(6.95, 3.05),
        gridspec_kw={"width_ratios": [1.18, 0.82]},
        constrained_layout=True,
    )

    static_rows: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for architecture, tag, tap in [
        ("MultVAE", "vae", "z"),
        ("NeuMF", "ncf", "user_state"),
    ]:
        for dataset_slug, dataset_label in [
            ("ml1m", "ML1M"),
            ("yahoo", "Yahoo"),
            ("pinterest", "Pinterest"),
        ]:
            frame = pd.read_csv(
                RESULTS / f"{dataset_slug}_{tag}_adversarial_k10.csv"
            )
            frame = frame.loc[frame["tap"] == tap].copy()
            methods = ("attack_lam0", "attack_lam1")
            common_users = set.intersection(*(
                set(frame.loc[frame["method"] == method, "user"])
                for method in methods
            ))
            counts = []
            for method in methods:
                cohort = frame.loc[
                    (frame["method"] == method)
                    & frame["user"].isin(common_users)
                ]
                if cohort["user"].duplicated().any():
                    raise ValueError(
                        f"duplicate users for {architecture}, {dataset_label}, "
                        f"{method}"
                    )
                counts.append(_joint_tail_count(cohort))
            static_rows.append(
                (f"{architecture} / {dataset_label}", counts[0], counts[1])
            )

    y_positions = np.arange(len(static_rows))[::-1]
    behavior_color = "#0072B2"
    whitebox_color = "#D55E00"
    for y, (label, behavior, whitebox) in zip(y_positions, static_rows):
        behavior_rate = 100.0 * behavior[0] / behavior[1]
        whitebox_rate = 100.0 * whitebox[0] / whitebox[1]
        static_ax.plot(
            [behavior_rate, whitebox_rate], [y, y],
            color="#b8bdc5", linewidth=1.2, zorder=1,
        )
        for rate, count, offset, color, marker, legend_label in [
            (behavior_rate, behavior, 0.11, behavior_color, "o",
             r"behavior-only ($\lambda=0$)"),
            (whitebox_rate, whitebox, -0.11, whitebox_color, "s",
             r"white-box ($\lambda=1$)"),
        ]:
            lower, upper = _wilson_interval(*count)
            static_ax.errorbar(
                rate, y + offset,
                xerr=[[rate - lower], [upper - rate]],
                fmt=marker, markersize=4.0, markeredgewidth=0.4,
                markeredgecolor="white", color=color, ecolor=color,
                elinewidth=0.8, capsize=1.8, zorder=3,
                label=legend_label if y == y_positions[0] else None,
            )
            static_ax.text(
                upper + 0.8, y + offset, _rate_label(rate),
                va="center", ha="left", fontsize=5.9, color=color,
            )

    static_ax.axhline(2.5, color="#d4d7dc", linewidth=0.7)
    static_ax.set_yticks(y_positions, [row[0] for row in static_rows])
    static_ax.set_xlim(-2.5, 88)
    static_ax.set_xticks([0, 20, 40, 60, 80])
    static_ax.set_ylim(-0.6, 5.65)
    static_ax.set_xlabel("joint-tail rate (% of paired cohort)")
    static_ax.set_title("(a) Static recommenders, $K=10$", loc="left",
                        fontweight="bold")
    static_ax.grid(axis="x", color="#e0e2e6", linewidth=0.5, zorder=0)
    static_ax.legend(loc="lower right", frameon=False, handletextpad=0.3,
                     borderaxespad=0.2)

    ordinary = pd.read_csv(RESULTS / "ml1m_sasrec_explainers_k5.csv")
    ordinary = ordinary.loc[ordinary["tap"] == "block1"].copy()
    whitebox = pd.read_csv(RESULTS / "ml1m_sasrec_attack_k5.csv")
    whitebox = whitebox.loc[whitebox["tap"] == "block1"].copy()
    sasrec_rows = [
        ("White-box search", whitebox.loc[
            whitebox["explainer"] == "attack_lam1"], whitebox_color),
        ("Integrated gradients", ordinary.loc[
            ordinary["explainer"] == "ig"], behavior_color),
        ("LIME", ordinary.loc[
            ordinary["explainer"] == "lime"], behavior_color),
        ("Occlusion", ordinary.loc[
            ordinary["explainer"] == "occlusion"], behavior_color),
        ("Saliency", ordinary.loc[
            ordinary["explainer"] == "saliency"], behavior_color),
        ("Random", ordinary.loc[
            ordinary["explainer"] == "random"], "#777777"),
        ("Popularity", ordinary.loc[
            ordinary["explainer"] == "pop"], "#777777"),
    ]
    sasrec_y = np.arange(len(sasrec_rows))[::-1]
    for y, (label, cohort, color) in zip(sasrec_y, sasrec_rows):
        count = _joint_tail_count(cohort)
        rate = 100.0 * count[0] / count[1]
        lower, upper = _wilson_interval(*count)
        sasrec_ax.errorbar(
            rate, y, xerr=[[rate - lower], [upper - rate]], fmt="o",
            markersize=4.2, markeredgewidth=0.4, markeredgecolor="white",
            color=color, ecolor=color, elinewidth=0.85, capsize=2.0, zorder=3,
        )
        sasrec_ax.text(upper + 0.8, y, _rate_label(rate), va="center", ha="left",
                       fontsize=5.9, color=color)

    sasrec_ax.axhline(5.5, color="#d4d7dc", linewidth=0.7)
    sasrec_ax.set_yticks(sasrec_y, [row[0] for row in sasrec_rows])
    sasrec_ax.set_xlim(-1.5, 56)
    sasrec_ax.set_xticks([0, 10, 20, 30, 40, 50])
    sasrec_ax.set_ylim(-0.6, 6.6)
    sasrec_ax.set_xlabel("joint-tail rate (% of cohort)")
    sasrec_ax.set_title("(b) SASRec, ML1M ($K=5$, block 1)", loc="left",
                        fontweight="bold")
    sasrec_ax.grid(axis="x", color="#e0e2e6", linewidth=0.5, zorder=0)

    for ax in (static_ax, sasrec_ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results_overview.pdf"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def _static_mechanism_groups(
    architecture: str,
) -> list[tuple[str, pd.DataFrame, str]]:
    if architecture == "MultVAE":
        tag, tap, exact_column = "vae", "z", "switch_mass"
    elif architecture == "NeuMF":
        tag, tap, exact_column = "ncf", "user_state", "gate_mass"
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

    groups = []
    for dataset_slug, dataset_label in [
        ("ml1m", "ML1M"),
        ("yahoo", "Yahoo"),
        ("pinterest", "Pinterest"),
    ]:
        audit = pd.read_csv(
            RESULTS / f"{dataset_slug}_{tag}_adversarial_k10.csv"
        )
        audit = audit.loc[
            (audit["method"] == "attack_lam1") & (audit["tap"] == tap),
            ["user", "delta_beh", "jcomp"],
        ]
        exact = pd.read_csv(
            RESULTS / f"{dataset_slug}_{tag}_switchdiag_k10.csv"
        )
        exact = exact.loc[
            exact["method"] == "attack_lam1",
            ["user", "delta_beh", exact_column],
        ]
        frame = audit.merge(
            exact, on="user", suffixes=("_audit", "_exact"),
            validate="one_to_one",
        )
        if not np.allclose(
            frame["delta_beh_audit"], frame["delta_beh_exact"],
            atol=1e-7, rtol=0,
        ):
            raise ValueError(
                f"effect mismatch for {architecture}, {dataset_label}"
            )
        frame = frame.loc[frame["delta_beh_audit"] > 0.01].copy()
        frame["audit_remainder"] = (
            frame["delta_beh_audit"] - frame["jcomp"]
        )
        frame["exact_effect"] = frame[exact_column]
        groups.append((dataset_label, frame, DATASET_COLORS[dataset_label]))
    return groups


def _sasrec_mechanism_groups() -> list[tuple[str, pd.DataFrame, str]]:
    groups = []
    method_colors = {
        "LIME": "#0072B2",
        "Occlusion": "#009E73",
        "white-box": "#D55E00",
    }
    for filename in [
        "ml1m_sasrec_basis_robustness_k5_s42_carrier.csv",
        "ml1m_sasrec_basis_robustness_k5_attack_s42_carrier.csv",
    ]:
        frame = pd.read_csv(RESULTS / filename)
        frame["clean_gap"] = np.clip(
            frame["clean_unrecovered"] / frame["delta_beh"], 0.0, 1.0
        )
        frame = frame.loc[
            (frame["delta_beh"] > 0.2) & (frame["clean_gap"] > 0.5)
        ].copy()
        frame["audit_remainder"] = frame["clean_unrecovered"]
        frame["exact_effect"] = (
            frame["attention_context_mass"] + frame["ffn_mass"]
        )
        for method, subset in frame.groupby("method"):
            label = {
                "lime": "LIME",
                "occlusion": "Occlusion",
                "attack_lam1": "white-box",
            }[method]
            groups.append((label, subset, method_colors[label]))
    return groups


def _mechanism_scatter(
    ax: mpl.axes.Axes,
    groups: list[tuple[str, pd.DataFrame, str]],
    title: str,
) -> None:
    low = 0.0
    high = 0.0
    for label, frame, color in groups:
        rho = frame["audit_remainder"].corr(
            frame["exact_effect"], method="spearman"
        )
        ax.scatter(
            frame["audit_remainder"], frame["exact_effect"],
            s=10, alpha=0.48, linewidths=0, color=color,
            label=fr"{label}  {rho:.2f}", rasterized=False,
        )
        low = min(low, float(frame[["audit_remainder", "exact_effect"]].min().min()))
        high = max(high, float(frame[["audit_remainder", "exact_effect"]].max().max()))

    span = high - low
    lower = low - 0.035 * span
    upper = high + 0.055 * span
    ax.plot([lower, upper], [lower, upper], color="#555555", linewidth=0.75,
            linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"audit remainder $U$")
    ax.set_ylabel("independently decomposed effect")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, color="#e1e3e6", linewidth=0.4, zorder=0)
    legend = ax.legend(
        loc="upper left", frameon=False, handletextpad=0.25,
        borderaxespad=0.2, labelspacing=0.2, title=r"group  $\rho_s$",
    )
    legend.get_title().set_fontsize(6.1)


def mechanism_validation() -> Path:
    """Validate the audit remainder against exact mechanism effects."""
    _set_style()
    fig, (vae_ax, ncf_ax, sasrec_ax) = plt.subplots(
        1, 3, figsize=(6.95, 2.45), constrained_layout=True
    )

    _mechanism_scatter(
        vae_ax, _static_mechanism_groups("MultVAE"),
        "(a) MultVAE: opened ReLUs",
    )
    _mechanism_scatter(
        ncf_ax, _static_mechanism_groups("NeuMF"),
        "(b) NeuMF: gate transitions",
    )
    _mechanism_scatter(
        sasrec_ax, _sasrec_mechanism_groups(),
        "(c) SASRec: block deviations",
    )
    ncf_ax.set_ylabel("")
    sasrec_ax.set_ylabel("")

    for ax in (vae_ax, ncf_ax, sasrec_ax):
        for spine in ax.spines.values():
            spine.set_color("#777777")

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "carrier_validation.pdf"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> int:
    targets = [headline_results(), results_overview(), mechanism_validation()]
    for target in targets:
        print(target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
