"""Selection-safe aggregations used by the manuscript tables.

The paper compares methods on paired user cohorts.  Behavioral means are
therefore computed on the intersection of users available for every requested
method, before any effect-size filter is applied.  The ``delta_min`` filter is
used only for ratio metrics, whose denominators are unstable near zero.

The helpers also upgrade early result files that predate the additive
two-order mediation columns. In those files ``jrestore`` stores the reverse
reproduced-effect ratio. The legacy ratio columns are retained only for
backward compatibility; manuscript summaries use raw effects and bounded
mechanism gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _augment_mediation(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the additive mediation columns filled consistently.

    Current artifacts contain raw forward/reverse effects; older artifacts
    contain only ``delta_beh``, ``jcomp``, ``mr``, and the reverse ratio named
    ``jrestore``.  Both formats are supported without mutating the caller's
    DataFrame.
    """
    out = df.copy()
    required = {"delta_beh", "jcomp", "mr", "jrestore"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"missing mediation columns: {sorted(missing)}")

    # Early CSVs store the reverse recovery ratio in ``jrestore``.  Recover
    # its raw marginal effect when it is not already present.
    if "jrestore_raw" not in out:
        out["jrestore_raw"] = out["jrestore"] * out["delta_beh"]
    else:
        out["jrestore_raw"] = out["jrestore_raw"].fillna(
            out["jrestore"] * out["delta_beh"]
        )

    out["jshap"] = 0.5 * (out["jcomp"] + out["jrestore_raw"])
    # Use the ratio identity rather than re-dividing by Delta.  This preserves
    # the exact order average encoded in old files, including their epsilon
    # guard convention.
    out["smr"] = 0.5 * (out["mr"] + out["jrestore"])
    out["jshap_complement"] = out["delta_beh"] - out["jshap"]
    out["clean_unrecovered"] = out["delta_beh"] - out["jcomp"]
    positive = out["delta_beh"] > 1e-6
    out["clean_gap"] = np.nan
    out.loc[positive, "clean_gap"] = np.clip(
        out.loc[positive, "clean_unrecovered"]
        / out.loc[positive, "delta_beh"], 0.0, 1.0)
    out["symmetric_gap"] = np.nan
    out.loc[positive, "symmetric_gap"] = np.clip(
        out.loc[positive, "jshap_complement"]
        / out.loc[positive, "delta_beh"], 0.0, 1.0)
    if "selectivity_del" in out:
        selectivity = out["selectivity_del"].clip(lower=0)
        out["target_share_del"] = selectivity / (1.0 + selectivity)
    out["interaction_gap"] = out["jcomp"] - out["jrestore_raw"]
    return out


def _common_users(frame: pd.DataFrame, methods: Iterable[str]) -> frozenset:
    methods = list(methods)
    if not methods:
        raise ValueError("at least one method is required")
    available = set(frame["method"].dropna().unique())
    missing = set(methods).difference(available)
    if missing:
        raise ValueError(f"methods absent from artifact: {sorted(missing)}")
    user_sets = [set(frame.loc[frame["method"] == method, "user"])
                 for method in methods]
    return frozenset.intersection(*(frozenset(s) for s in user_sets))


def paired_attack_summary(
    df: pd.DataFrame,
    methods: Iterable[str],
    *,
    tap: str = "z",
    delta_min: float = 0.01,
    success_delta: float = 0.2,
    success_gap: float = 0.5,
) -> dict[str, dict[str, object]]:
    """Summarize attack methods on one identical, paired user cohort.

    ``delta_mean`` and success rates are unconditional over the common cohort.
    Gap medians alone use ``delta_beh > delta_min``. Duplicate
    method/user rows at a tap are rejected because they would silently change
    the weighting of the paper table.
    """
    methods = list(methods)
    frame = _augment_mediation(df)
    frame = frame.loc[frame["tap"] == tap].copy()
    if frame.empty:
        raise ValueError(f"tap {tap!r} is absent from artifact")

    common = _common_users(frame, methods)
    if not common:
        raise ValueError("requested methods have no users in common")
    paired = frame.loc[
        frame["method"].isin(methods) & frame["user"].isin(common)
    ]
    duplicate = paired.duplicated(["method", "user"], keep=False)
    if duplicate.any():
        keys = paired.loc[duplicate, ["method", "user"]].drop_duplicates()
        raise ValueError(f"duplicate method/user rows at tap {tap!r}: "
                         f"{keys.to_dict('records')[:5]}")

    out: dict[str, dict[str, object]] = {}
    for method in methods:
        group = paired.loc[paired["method"] == method]
        if len(group) != len(common):
            raise ValueError(f"method {method!r} is not one-to-one on cohort")
        positive = group.loc[group["delta_beh"] > delta_min]
        success = ((group["delta_beh"] > success_delta)
                   & (group["clean_gap"] > success_gap))
        out[method] = {
            "users": common,
            "n_users": len(common),
            "n_ratio": len(positive),
            "delta_mean": float(group["delta_beh"].mean()),
            "mr_median": float(positive["mr"].median()),
            "smr_median": float(positive["smr"].median()),
            "clean_gap_median": float(positive["clean_gap"].median()),
            "clean_gap_p90": float(positive["clean_gap"].quantile(0.9)),
            "symmetric_gap_median": float(
                positive["symmetric_gap"].median()),
            "success_count": int(success.sum()),
            "success_rate": float(success.mean()),
        }

    reference = out[methods[0]]["delta_mean"]
    for method in methods:
        value = out[method]["delta_mean"]
        out[method]["delta_relative"] = (
            float(value / reference) if not np.isclose(reference, 0.0)
            else float("nan")
        )
    return out


def _json_ready(summary: dict[str, dict[str, object]]) -> dict:
    # User IDs are retained by ``paired_attack_summary`` for programmatic
    # pairing checks, but add no value to the command-line report.
    return {
        method: {key: value for key, value in values.items()
                 if key != "users"}
        for method, values in summary.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a paired stress-test summary from a result CSV"
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--methods", nargs="+", default=[
        "saliency", "attack_lam1", "constraint"
    ])
    parser.add_argument("--tap", default="z")
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.csv)
    print(json.dumps(_json_ready(paired_attack_summary(
        frame, args.methods, tap=args.tap
    )), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
