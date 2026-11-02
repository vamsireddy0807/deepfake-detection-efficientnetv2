"""Evaluate on the held-out test split, with and without horizontal-flip TTA.

    python scripts/evaluate.py --root data/real_vs_fake/real-vs-fake \
        --model artifacts/effv2b0_deepfake_best.keras --out artifacts/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score  # noqa: E402

from deepfake_detection.data import index_split, make_ds, make_flip_ds  # noqa: E402
from deepfake_detection.model import WarmupCosine  # noqa: E402,F401  (registers the schedule for loading)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--out", type=Path, default=Path("test_metrics.json"))
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    df = index_split(args.root, args.split)
    y = df.label.values

    p_single = model.predict(make_ds(df), verbose=1).ravel()
    p_flip = model.predict(make_flip_ds(df), verbose=1).ravel()
    p_tta = (p_single + p_flip) / 2.0

    def block(p):
        pred = (p >= 0.5).astype(int)
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "report": classification_report(y, pred, target_names=["fake", "real"], output_dict=True, digits=4),
        }

    out = {"split": args.split, "images": int(len(df)), "single_pass": block(p_single), "tta_hflip": block(p_tta)}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"single-pass acc {out['single_pass']['accuracy']} auc {out['single_pass']['auc']}")
    print(f"TTA (x2)    acc {out['tta_hflip']['accuracy']} auc {out['tta_hflip']['auc']}")


if __name__ == "__main__":
    main()
