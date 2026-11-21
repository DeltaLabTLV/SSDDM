"""Train SASRec seed replicates with the vendored LXSR pipeline.

The manuscript's sequential result rests on one SASRec checkpoint, so its
strong-gap rate for ordinary explainers is a single-checkpoint observation.
This trains additional checkpoints that differ only in the training seed, on
the identical GTS q=0.8 Movielens-1m split, so the rate can be reported with a
spread rather than as a point.

Two behaviours of the vendored trainer need patching:

* ``Trainer._set_seed`` hard-codes 42, so replicates would be identical;
* ``Trainer.log_best_model`` deletes the checkpoint from disk after staging it
  into a Weights & Biases artifact, which loses the file in offline mode.

Both are patched here rather than in the vendored tree, which stays pristine.

Usage:
  python3 train_sasrec_seeds.py --seeds 42 7 123
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LXSR = ROOT / "external" / "LXSR"

# The vendored pipeline reads its data root from the environment and imports
# `src.*` relative to its own checkout.
os.environ.setdefault("SEQ_SPLITS_DATA_PATH", str(LXSR / "data"))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
sys.path.insert(0, str(LXSR))

# Matches the audited checkpoint: 2 blocks, d=64, tied embeddings, reverse
# positional direction (the property the position-freezing control relies on).
BASE_OVERRIDES = [
    "model.model_type=sasrec",
    "model.d_model=64",
    "model.n_head=2",
    "model.num_layers=2",
    "model.dropout=0.2",
    "model.attention_dropout=0.2",
    "model.layer_norm_eps=1e-6",
    "model.embedding_type=tied",
    "model.positional_direction=reverse",
    "model.max_seq_length=200",
    "dataloader.batch_size=256",
    "dataloader.num_workers=2",
    "training.learning_rate=0.001",
    "training.weight_decay=0.0",
    "training.label_smoothing=0.0",
    "training.num_epochs=100",
    "training.patience=10",
    "training.eval_mode=successive",
    "training.distributed=false",
    "wandb.use_wandb=true",  # offline; the trainer needs a run id
]


_ORIGINALS: dict = {}


def patch_trainer(seed: int, keep_path: Path):
    """Make the seed configurable and stop the trainer deleting the weights.

    The pristine methods are captured once. Re-patching a class that already
    carries a wrapper would otherwise chain the wrappers, so every earlier
    seed's ``keep_path`` would be rewritten with the newest seed's weights --
    silently producing identical replicates.
    """
    import random

    import numpy as np
    import torch
    from src.recommenders.trainer import Trainer

    if not _ORIGINALS:
        _ORIGINALS["evaluate"] = Trainer.evaluate
        _ORIGINALS["log_best_model"] = Trainer.log_best_model

    def _set_seed(self):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # The trainer only ships its final metrics to wandb, so capture every
    # evaluate() result; the last one is the final test evaluation.
    original_evaluate = _ORIGINALS["evaluate"]

    def evaluate(self, eval_loader):
        result = original_evaluate(self, eval_loader)
        if isinstance(result, dict):
            self.last_metrics = {k: float(v) for k, v in result.items()}
        return result

    original_log = _ORIGINALS["log_best_model"]

    def log_best_model(self):
        src = Path(self.best_model_path)
        if src.exists():
            keep_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, keep_path)
            tok = keep_path.with_suffix(".tokenizer.json")
            tok.write_text(json.dumps(self.tokenizer.to_json()))
            print(f"  kept checkpoint -> {keep_path}", flush=True)
        return original_log(self)

    Trainer._set_seed = _set_seed
    Trainer.evaluate = evaluate
    Trainer.log_best_model = log_best_model
    return Trainer


def train_one(dataset: str, seed: int, out_dir: Path, extra):
    keep_path = out_dir / f"sasrec_{dataset.lower()}_s{seed}.pth"
    if keep_path.exists():
        print(f"[skip] {keep_path.name} exists")
        return None

    from src.config_utils import build_config

    Trainer = patch_trainer(seed, keep_path)
    config = build_config(dataset, phase="final",
                          overrides=BASE_OVERRIDES + list(extra) + [
                              f"wandb.run_name=sasrec_s{seed}"])
    t0 = time.time()
    trainer = Trainer(config)
    trainer.train()
    minutes = (time.time() - t0) / 60.0

    # Each seed needs its own wandb run: the trainer derives best_model_path
    # from wandb.run.id, so a reused run would make every seed write to the
    # same file.
    import wandb
    if wandb.run is not None:
        wandb.finish()

    metrics = getattr(trainer, "last_metrics", {})
    record = {"dataset": dataset, "seed": seed, "minutes": minutes,
              "checkpoint": keep_path.name, "test_metrics": metrics}
    summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())
                        if "20" in k or "10" in k)
    print(f"[done] seed {seed} in {minutes:.1f} min -> {keep_path.name}"
          f"  {summary}", flush=True)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Movielens-1m")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--out-dir", default=str(ROOT / "check" / "sasrec_seeds"))
    ap.add_argument("--out", default=str(ROOT / "jlens_results" /
                                         "sasrec_seed_replicates.json"))
    ap.add_argument("overrides", nargs="*",
                    help="extra dotlist overrides for build_config")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_path = Path(args.out)
    records = json.loads(out_path.read_text()) if out_path.exists() else []

    for seed in args.seeds:
        record = train_one(args.dataset, seed, out_dir, args.overrides)
        if record is not None:
            records.append(record)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(records, indent=2))

    print(f"wrote {out_path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
