"""Two-phase training: frozen-backbone head (3 epochs), then top-30% fine-tune (7 epochs).

    kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data --unzip
    python scripts/train.py --root data/real_vs_fake/real-vs-fake --out artifacts/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf  # noqa: E402
from tensorflow.keras import callbacks, mixed_precision, optimizers  # noqa: E402

from deepfake_detection.data import BATCH_SIZE, IMG_SIZE, index_split, make_ds  # noqa: E402
from deepfake_detection.model import WarmupCosine, build_model, unfreeze_top  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="folder containing train/ valid/ test/")
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=7)
    ap.add_argument("--unfreeze", type=float, default=0.30, help="fraction of backbone layers to fine-tune")
    ap.add_argument("--no-mixed-precision", action="store_true")
    args = ap.parse_args()

    if not args.no_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = str(args.out / "effv2b0_deepfake_best.keras")

    train_df = index_split(args.root, "train")
    val_df = index_split(args.root, "valid")
    print(f"train {len(train_df)} | val {len(val_df)}")
    train_ds = make_ds(train_df, training=True)
    val_ds = make_ds(val_df)

    model, base = build_model()
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    # Phase 1: head only, higher learning rate
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=metrics)
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=[callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max", save_best_only=True)],
    )

    # Phase 2: unfreeze the top of the backbone, warm-up + cosine at 1e-4
    unfreeze_top(base, args.unfreeze)
    steps = math.ceil(len(train_df) / BATCH_SIZE)
    schedule = WarmupCosine(1e-4, warmup_steps=steps, total_steps=args.epochs_ft * steps)
    model.compile(
        optimizer=optimizers.Adam(schedule),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_ft,
        callbacks=[
            callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max", save_best_only=True),
            callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )

    history = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in h1.history}
    summary = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_head": args.epochs_head,
        "epochs_fine_tune": args.epochs_ft,
        "best_val_auc": max(history["val_auc"]),
        "final_val_accuracy": history["val_accuracy"][-1],
        "checkpoint": ckpt,
    }
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
