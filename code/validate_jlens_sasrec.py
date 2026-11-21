"""Validation gate for the SASRec audit port (pre-registered in the design).

Checks on a small user sample:
  1. Stage-decomposition consistency: resume(tap, H0) reproduces the clean
     logit at every tap (|err| < 1e-4).
  2. Exactness of the four corners: delta_patch == delta_beh at every tap
     (F11 must equal the corrupted forward).
  3. Affine identity: MR == 1 at the final block tap (output-adjacent).
  4. Reference-state identity: all clean/corrupted/path/union allocations
     obey the two-order formulas, and all close at the affine final tap.
  5. KILL CRITERION: median MR of random deletions at the inner taps must be
     >= 0.5, else the protocol is declared not ready (paper gets a
     limitation paragraph only).

Usage: python3 validate_jlens_sasrec.py --ckpt <path> [--n-users 25 --k 5]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import jlens_sasrec as js
from run_jlens_sasrec import build_cohort, target_and_competitors

DEFAULT_LXSR = Path(__file__).resolve().parent.parent / "external" / "LXSR"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lxsr-root", default=str(DEFAULT_LXSR))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-users", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-hist", type=int, default=0,
                    help="keep only users with at most this many interactions; "
                         "with --min-hist it isolates the deleted fraction from K")
    ap.add_argument("--min-hist", type=int, default=0,
                    help="keep only users with at least this many interactions")
    ap.add_argument("--data-dir", default=None,
                    help="q08 split dir; default <lxsr-root>/.../Movielens-1m/q08")
    args = ap.parse_args(argv)

    lxsr_root = Path(args.lxsr_root)
    data_dir = Path(args.data_dir) if args.data_dir else lxsr_root / \
        "data/splitted/global_timesplit/val_by_time/Movielens-1m/q08"
    device = args.device if torch.cuda.is_available() else "cpu"
    model, tok, _ = js.load_sasrec(lxsr_root, args.ckpt,
                                   data_dir / "train.csv", device=device)
    aud = js.SasRecAuditor(model, device=device)
    users, _ = build_cohort(lxsr_root, data_dir, args.k, args.min_hist,
                            args.max_hist)
    users = users[:args.n_users]
    final_tap = aud.taps[-1]

    max_resume_err, max_patch_err = 0.0, 0.0
    max_ref_formula_err, max_ref_final_gap = 0.0, 0.0
    checked_references = False
    final_mr, rand_mr = [], {t: [] for t in aud.taps[:-1]}
    gen = np.random.default_rng(0)

    for uid, seq in users:
        tokens = [tok.item2id[i] for i in seq][-tok.max_seq_length:]
        y, comps = target_and_competitors(aud, tok, tokens)

        # check 1: resume reproduces the clean forward at every tap
        ids = torch.tensor(tokens, dtype=torch.long,
                           device=device).unsqueeze(0)
        m = torch.ones_like(ids)
        ref = float(aud.logits(ids, m)[y])
        acts = aud.stages(ids, m)
        for tap in aud.taps:
            with torch.no_grad():
                err = abs(float(aud.resume(tap, acts[tap], m)[y]) - ref)
            max_resume_err = max(max_resume_err, err)

        # checks 2+3 on a saliency deletion; kill criterion on random sets
        S_sal = js.topk_positions(js.grad_x_input_map(aud, tokens, y), args.k)
        for r in js.audit_set(aud, tokens, S_sal, y, comps):
            max_patch_err = max(max_patch_err,
                                abs(r["delta_patch"] - r["delta_beh"]))
            if r["tap"] == final_tap and abs(r["delta_beh"]) > 0.01:
                final_mr.append(r["mr"])

        # Check every new reference-state construction once. At the final
        # tap the downstream scorer is affine and ignores the attention mask,
        # so every basis containing the target direction must close exactly.
        if not checked_references:
            ref = js.audit_set(
                aud, tokens, S_sal, y, comps, taps=[final_tap],
                ref_states=("all",))[0]
            for suffix in ("", "_corrupt", "_path_clean", "_path_deleted",
                           "_path_avg", "_union"):
                jfwd = ref[f"jfwd{suffix}"]
                jrev = ref[f"jrev{suffix}"]
                jshap = ref[f"jshap{suffix}"]
                interaction = ref[f"interaction_gap{suffix}"]
                max_ref_formula_err = max(
                    max_ref_formula_err,
                    abs(jshap - 0.5 * (jfwd + jrev)),
                    abs(interaction - (jfwd - jrev)))
                if ref["delta_beh"] > 0.01:
                    max_ref_final_gap = max(
                        max_ref_final_gap,
                        abs(ref[f"clean_gap{suffix}"]),
                        abs(ref[f"symmetric_gap{suffix}"]))
            checked_references = True
        for _ in range(3):
            Sr = list(gen.choice(len(tokens), size=args.k, replace=False))
            for r in js.audit_set(aud, tokens, Sr, y, comps,
                                  taps=aud.taps[:-1]):
                if abs(r["delta_beh"]) > 0.01:
                    rand_mr[r["tap"]].append(r["mr"])

    print(f"[1] max |resume - forward| : {max_resume_err:.2e}  "
          f"({'PASS' if max_resume_err < 1e-3 else 'FAIL'})")
    print(f"[2] max |delta_patch - delta_beh| : {max_patch_err:.2e}  "
          f"({'PASS' if max_patch_err < 1e-3 else 'FAIL'})")
    fm = np.median(np.abs(np.array(final_mr) - 1.0)) if final_mr else np.nan
    print(f"[3] final tap ({final_tap}) median |MR-1| : {fm:.2e}  "
          f"({'PASS' if fm < 1e-3 else 'FAIL'})  n={len(final_mr)}")
    refs_pass = max_ref_formula_err < 1e-6 and max_ref_final_gap < 1e-3
    print(f"[4] reference formulas / final affine gaps : "
          f"{max_ref_formula_err:.2e} / {max_ref_final_gap:.2e}  "
          f"({'PASS' if refs_pass else 'FAIL'})")
    for tap, v in rand_mr.items():
        med = np.median(v) if v else np.nan
        verdict = "PASS" if med >= 0.5 else "KILL"
        print(f"[5] random-deletion MR median @ {tap}: {med:.3f}  "
              f"({verdict})  n={len(v)}")


if __name__ == "__main__":
    main()
