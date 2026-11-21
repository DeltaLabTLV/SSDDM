"""SASRec (LXSR codebase) mechanistic audit on ML1M: explainer table + red-team.

Cohort mirrors eval_per_user.py: SessionDataset over the q08 GTS test.csv with
warm-item filtering; per user the input is session_items[:-1] (leave-one-out
input), y = top-ranked NON-INPUT item at the last position, competitors = the
next 5.  Explainers score POSITIONS of the input sequence; deletion is
delete-and-shift (LXSR pos_perturbation).

Usage:
  python3 run_jlens_sasrec.py --mode explainers --n-users 300 --k 5
  python3 run_jlens_sasrec.py --mode attack --n-users 100 --k 5 10
Writes <out-dir>/ml1m_sasrec_<mode>_k<K>.csv
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

DEFAULT_LXSR = Path(__file__).resolve().parent.parent / "external" / "LXSR"


def build_cohort(lxsr_root, data_dir, k_min, min_hist=0, max_hist=0):
    sys.path.insert(0, str(lxsr_root))
    from src.recommenders.dataset import SessionDataset
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    warm = train_df["item_id"].unique().tolist()
    ds = SessionDataset(df=test_df, session_key="user_id",
                        item_key="item_id", time_key="timestamp",
                        warm_items=warm)
    users = []
    for i in range(len(ds)):
        s = ds[i]
        seq = s["session_items"][:-1]  # leave-one-out input
        if (len(seq) > k_min + 1 and len(seq) >= min_hist
                and (max_hist == 0 or len(seq) <= max_hist)):
            users.append((int(s["session_id"]), seq))
    return users, train_df


def target_and_competitors(aud, tok, tokens, n_comp=5):
    ids = torch.tensor(tokens, dtype=torch.long,
                       device=aud.device).unsqueeze(0)
    logits = aud.logits(ids, torch.ones_like(ids)).clone()
    logits[0] = float("-inf")  # pad token
    logits[list(set(tokens))] = float("-inf")  # non-history targets only
    order = logits.argsort(descending=True)
    return int(order[0]), order[1:1 + n_comp].tolist()


def pop_map(tokens, pop_counts):
    return torch.tensor([pop_counts.get(t, 0) for t in tokens],
                        dtype=torch.float32)


# ----------------------------- red-team ------------------------------------

def search_score(row, score="delta_minus_j", lam=1.0):
    """Return the scalar optimized by one search mode."""
    delta, jfwd = float(row["delta_beh"]), float(row["jfwd"])
    if score == "jfwd":
        return jfwd
    if score == "delta_minus_j":
        return delta - lam * jfwd
    raise ValueError(f"unknown search score {score!r}")


def attack_label(score, lam=1.0):
    if score == "jfwd":
        return "attack_jfwd"
    if score == "delta_minus_j":
        return f"attack_lam{lam:g}"
    raise ValueError(f"unknown search score {score!r}")


def attack_objective(aud, tokens, S, y, comps, tap, lam=1.0,
                     score="delta_minus_j"):
    rows = js.audit_set(aud, tokens, S, y, comps, taps=[tap])
    r = rows[0]
    return search_score(r, score=score, lam=lam), r


def greedy_attack(aud, tokens, y, comps, k, tap="block1", lam=1.0,
                  score="delta_minus_j", pool_size=40, n_seeds=6,
                  refine_iters=15, seed=0):
    """Run the same attribution-seeded local search under a chosen scalar."""
    n = len(tokens)
    occ = js.occlusion_map(aud, tokens, y)
    pool = occ.abs().argsort(descending=True)[:min(pool_size, n)].tolist()
    gen = np.random.default_rng(seed)

    seeds = [occ.argsort(descending=True)[:k].tolist()]
    sal = js.grad_x_input_map(aud, tokens, y)
    seeds.append(sal.argsort(descending=True)[:k].tolist())
    for _ in range(n_seeds - len(seeds)):
        seeds.append(list(gen.choice(pool, size=min(k, len(pool)),
                                     replace=False)))

    best_S, best_obj, best_row = None, -1e18, None
    for S in seeds:
        obj, row = attack_objective(
            aud, tokens, S, y, comps, tap, lam=lam, score=score)
        if obj > best_obj:
            best_S, best_obj, best_row = list(S), obj, row

    for _ in range(refine_iters):
        improved = False
        out_i = gen.integers(0, k)
        for cand in pool:
            if cand in best_S:
                continue
            S2 = list(best_S)
            S2[out_i] = cand
            obj, row = attack_objective(
                aud, tokens, S2, y, comps, tap, lam=lam, score=score)
            if obj > best_obj:
                best_S, best_obj, best_row = S2, obj, row
                improved = True
        if not improved:
            break
    return best_S, best_row


# ----------------------------- main ----------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lxsr-root", default=str(DEFAULT_LXSR))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="q08 split dir; default <lxsr-root>/data/.../Movielens-1m/q08")
    ap.add_argument("--mode", choices=["explainers", "attack"],
                    default="explainers")
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--k", type=int, nargs="+", default=[5])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--score", choices=["delta_minus_j", "jfwd"],
                    default="delta_minus_j",
                    help="search scalar: delta_minus_j gives max-Delta at "
                         "lam=0 and max-U at lam=1; jfwd gives max-J")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="",
                    help="suffix for the output CSV name, e.g. _s7")
    ap.add_argument("--prefix", default="ml1m",
                    help="dataset prefix for the output CSV name")
    ap.add_argument("--max-hist", type=int, default=0,
                    help="keep only users with at most this many interactions; "
                         "with --min-hist it isolates the deleted fraction from K")
    ap.add_argument("--min-hist", type=int, default=0,
                    help="keep only users with at least this many interactions; "
                         "used to hold the deleted fraction K/len comparable "
                         "across datasets with different sequence lengths")
    args = ap.parse_args(argv)

    lxsr_root = Path(args.lxsr_root)
    data_dir = Path(args.data_dir) if args.data_dir else \
        lxsr_root / "data/splitted/global_timesplit/val_by_time/Movielens-1m/q08"
    device = args.device if torch.cuda.is_available() else "cpu"

    model, tok, train_df = js.load_sasrec(
        lxsr_root, args.ckpt, data_dir / "train.csv", device=device)
    aud = js.SasRecAuditor(model, device=device)
    pop_counts = train_df["item_id"].map(tok.item2id).value_counts().to_dict()

    k_min = max(args.k)
    users, _ = build_cohort(lxsr_root, data_dir, k_min, args.min_hist,
                           args.max_hist)
    users = users[:args.n_users]
    lens = np.array([len(s) for _, s in users])
    print(f"cohort: {len(users)} users (hist > {k_min + 1}"
          f"{f', >= {args.min_hist}' if args.min_hist else ''}); "
          f"median hist {np.median(lens):.0f}, "
          f"median deleted fraction {np.median(k_min / lens):.3f}", flush=True)

    for k in args.k:
        rows = []
        t0 = time.time()
        for pos, (uid, seq) in enumerate(users):
            tokens = [tok.item2id[i] for i in seq][-tok.max_seq_length:]
            y, comps = target_and_competitors(aud, tok, tokens)
            if args.mode == "explainers":
                expl = {
                    "saliency": js.topk_positions(
                        js.grad_x_input_map(aud, tokens, y), k),
                    "ig": js.topk_positions(
                        js.grad_x_input_map(aud, tokens, y, steps=20), k),
                    "occlusion": js.topk_positions(
                        js.occlusion_map(aud, tokens, y), k),
                    "lime": js.topk_positions(
                        js.lime_map(aud, tokens, y, seed=uid), k),
                    "pop": js.topk_positions(pop_map(tokens, pop_counts), k),
                    "random": list(np.random.default_rng(uid)
                                   .choice(len(tokens), size=k,
                                           replace=False)),
                }
                for name, S in expl.items():
                    for r in js.audit_set(aud, tokens, S, y, comps,
                                          ref_states=("clean", "corrupted")):
                        rows.append({"user": uid, "explainer": name, **r})
            else:
                S, row = greedy_attack(aud, tokens, y, comps, k,
                                       lam=args.lam, score=args.score,
                                       seed=uid)
                for r in js.audit_set(aud, tokens, S, y, comps,
                                      ref_states=("clean", "corrupted")):
                    rows.append({"user": uid,
                                 "explainer": attack_label(args.score, args.lam),
                                 "S": json.dumps([int(s) for s in S]), **r})
            if (pos + 1) % 20 == 0:
                el = time.time() - t0
                print(f"K={k}: {pos + 1}/{len(users)} users "
                      f"({el / (pos + 1):.1f}s/user)", flush=True)

        df = pd.DataFrame(rows)
        out = (Path(args.out_dir)
               / f"{args.prefix}_sasrec_{args.mode}_k{k}{args.tag}.csv")
        df.to_csv(out, index=False)
        print(f"saved {len(df)} rows -> {out}")

        strong = df[(df.delta_beh > 0.01)]
        for e in sorted(df.explainer.unique()):
            for tap in aud.taps:
                z = strong[(strong.explainer == e) & (strong.tap == tap)]
                if not len(z):
                    continue
                print(f"K={k} {e:12s} {tap:7s}: n={z.user.nunique():3d} "
                      f"Del {df[(df.explainer == e) & (df.tap == tap)].delta_beh.mean():6.2f} "
                      f"MR med {z.mr.median():5.2f} q10 {z.mr.quantile(0.1):5.2f} "
                      f"SMR {z.smr.median():5.2f} "
                      f"MRcorr {z.mr_corr.median():5.2f} "
                      f"Sel {z.selectivity_del.median():5.2f}")


if __name__ == "__main__":
    main()
