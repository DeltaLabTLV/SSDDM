"""Positional recoding vs unit switching on SASRec (ML1M).

With positional_direction="reverse", position codes depend only on the slot,
so a mask-in-place deletion (token -> pad, attention mask zeroed, no shift)
removes content while FREEZING every surviving item's positional code; the
field's delete-and-shift protocol removes the same content AND repositions
the survivors.  Both are ordinary forward passes.  Auditing each corruption
separately therefore splits the SASRec low-recovery signal into a
positional-recoding component and a content/unit component:

  delta_shift = delta_mask + (s_mask - s_shift)   [recoding component]
  MRf_mask ~ 1  and  MRf_shift low  -> repositioning carries the instability
  MRf_mask low as well              -> content removal itself switches units

Sets per user: occlusion top-K, LIME top-K, random-K (explainer cohort, 300
users) and the saved white-box sets from ml1m_sasrec_attack_k{5,10}.csv
(search cohort, 100 users).

Usage (on the training host):
  python3 run_jlens_sasrec_maskinplace.py --ckpt <path> [--n-users 300]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import jlens_sasrec as js
from jlens_sasrec import EPS, delete_and_shift, orthonormal_basis
from run_jlens_sasrec import build_cohort, target_and_competitors

DEFAULT_LXSR = Path(__file__).resolve().parent.parent / "external" / "LXSR"


def mask_in_place(tokens, positions):
    out = torch.tensor(tokens, dtype=torch.long).clone()
    out[list(positions)] = 0
    return out


def audit_corruption(aud, tokens, S, y, competitors, corrupt, taps):
    """audit_set with a pluggable corruption operator."""
    device = aud.device
    ids0 = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    m0 = torch.ones_like(ids0)
    ids1 = corrupt(tokens, S).to(device).unsqueeze(0)
    m1 = (ids1 != 0).long()

    l0, l1 = aud.logits(ids0, m0), aud.logits(ids1, m1)
    delta_beh = float(l0[y] - l1[y])
    coll = np.mean([abs(float(l0[c] - l1[c])) for c in competitors])
    sel = abs(delta_beh) / max(coll, EPS)

    A0, A1 = aud.stages(ids0, m0), aud.stages(ids1, m1)
    rows = []
    for tap in taps:
        H0, H1 = A0[tap], A1[tap]
        dlt = (H1 - H0).flatten()
        V = aud.readout_directions(tap, H0, m0, [y] + list(competitors))
        Q = orthonormal_basis(V)
        pdlt = (Q @ (Q.T @ dlt)).view_as(H0)
        with torch.no_grad():
            F00 = float(aud.resume(tap, H0, m0)[y])
            F11 = float(aud.resume(tap, H1, m1)[y])
            F10 = float(aud.resume(tap, H0 + pdlt, m0)[y])
            F01 = float(aud.resume(tap, H1 - pdlt, m1)[y])
        denom = delta_beh if abs(delta_beh) > EPS else EPS
        rows.append({"tap": tap, "delta_beh": delta_beh,
                     "s_corr": float(l1[y]),
                     "jfwd": F00 - F10, "jrev": F01 - F11,
                     "mr": (F00 - F10) / denom,
                     "smr": 0.5 * ((F00 - F10) + (F01 - F11)) / denom,
                     "selectivity_del": sel, "k": len(S),
                     "hist_len": len(tokens)})
    return rows


def run_sets(aud, tokens, sets, y, comps, taps, rows, uid):
    for name, S in sets.items():
        for corr_name, corr in [("shift", delete_and_shift),
                                ("mask", mask_in_place)]:
            for r in audit_corruption(aud, tokens, S, y, comps, corr, taps):
                rows.append({"user": uid, "explainer": name,
                             "corruption": corr_name,
                             "S": json.dumps(sorted(int(s) for s in S)),
                             **r})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lxsr-root", default=str(DEFAULT_LXSR))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--attack-csv", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)

    lxsr_root = Path(args.lxsr_root)
    data_dir = Path(args.data_dir) if args.data_dir else \
        lxsr_root / "data/splitted/global_timesplit/val_by_time/Movielens-1m/q08"
    device = args.device if torch.cuda.is_available() else "cpu"
    model, tok, _ = js.load_sasrec(lxsr_root, args.ckpt,
                                   data_dir / "train.csv", device=device)
    aud = js.SasRecAuditor(model, device=device)
    taps = ["emb", "block1"]

    users, _ = build_cohort(lxsr_root, data_dir, args.k + 1)
    users = users[:args.n_users]
    print(f"cohort: {len(users)} users", flush=True)

    rows = []
    t0 = time.time()
    for i, (uid, seq) in enumerate(users):
        tokens = [tok.item2id[i2] for i2 in seq][-tok.max_seq_length:]
        y, comps = target_and_competitors(aud, tok, tokens)
        occ = js.occlusion_map(aud, tokens, y)
        lime = js.lime_map(aud, tokens, y, seed=uid)
        gen = np.random.default_rng(uid)
        sets = {
            "occlusion": js.topk_positions(occ, args.k),
            "lime": js.topk_positions(lime, args.k),
            "random": list(gen.choice(len(tokens), size=args.k,
                                      replace=False)),
        }
        run_sets(aud, tokens, sets, y, comps, taps, rows, uid)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(users)} ({(time.time()-t0)/(i+1):.1f}s/u)",
                  flush=True)

    out = Path(args.out_dir) / f"ml1m_sasrec_maskinplace_k{args.k}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("checkpointed explainer rows ->", out, flush=True)

    # saved white-box sets
    cohort_users, _ = build_cohort(lxsr_root, data_dir, 5)
    cohort = {uid: seq for uid, seq in cohort_users}
    for csv in args.attack_csv:
        adv = pd.read_csv(csv)
        adv = adv[adv.explainer.str.startswith("attack") &
                  (adv.tap == "block1")]
        for _, row in adv.iterrows():
            uid = int(row.user)
            if uid not in cohort:
                continue
            tokens = [tok.item2id[i2] for i2 in cohort[uid]
                      ][-tok.max_seq_length:]
            y, comps = target_and_competitors(aud, tok, tokens)
            S = json.loads(row.S)
            run_sets(aud, tokens, {f"attack_k{int(row.k)}": S}, y, comps,
                     taps, rows, uid)
        print(f"done {csv}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("wrote", out, flush=True)

    b1 = df[df.tap == "block1"]
    for name in b1.explainer.unique():
        sub = b1[b1.explainer == name]
        sh = sub[sub.corruption == "shift"].set_index("user")
        mk = sub[sub.corruption == "mask"].set_index("user")
        common = sh.index.intersection(mk.index)
        sh, mk = sh.loc[common], mk.loc[common]
        ok_s, ok_m = sh[sh.delta_beh > 0.01], mk[mk.delta_beh > 0.01]
        strong_s = ((sh.delta_beh > 0.2) & (sh.mr < 0.5)).mean()
        strong_m = ((mk.delta_beh > 0.2) & (mk.mr < 0.5)).mean()
        recod = ((sh.delta_beh - mk.delta_beh) / sh.delta_beh)[
            sh.delta_beh > 0.2]
        print(f"{name:>12}: d_shift {sh.delta_beh.median():.2f} "
              f"d_mask {mk.delta_beh.median():.2f} | "
              f"MRf shift {ok_s.mr.median():.2f} mask {ok_m.mr.median():.2f} | "
              f"strong shift {100*strong_s:.0f}% mask {100*strong_m:.0f}% | "
              f"recoding share med {recod.median():.2f} (n={len(recod)})",
              flush=True)


if __name__ == "__main__":
    main()
