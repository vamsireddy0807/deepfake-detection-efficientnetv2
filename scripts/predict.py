"""Classify a face image as REAL or FAKE and save a forensic explanation panel.

    python scripts/predict.py --model artifacts/effv2b0_deepfake_best.keras --image face.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from deepfake_detection.data import IMG_SIZE  # noqa: E402
from deepfake_detection.forensics import OPERATORS  # noqa: E402
from deepfake_detection.model import WarmupCosine  # noqa: E402,F401


def predict(model, image_path: str, tta: bool = True) -> dict:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = tf.cast(rgb[np.newaxis, ...], tf.float32)  # model rescales internally
    p = float(model.predict(x, verbose=0).squeeze())
    if tta:
        p = (p + float(model.predict(tf.image.flip_left_right(x), verbose=0).squeeze())) / 2.0
    label = "REAL" if p >= 0.5 else "FAKE"
    return {"label": label, "prob_real": round(p, 4), "confidence": round(p if p >= 0.5 else 1 - p, 4), "bgr": bgr}


def save_panel(result: dict, out_path: Path) -> None:
    bgr = result["bgr"]
    fig, axes = plt.subplots(1, 1 + len(OPERATORS), figsize=(4 * (1 + len(OPERATORS)), 4.2))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input")
    for ax, (name, op) in zip(axes[1:], OPERATORS.items(), strict=True):
        out = op(bgr)
        ax.imshow(out, cmap=None if out.ndim == 3 else "inferno")
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    color = "green" if result["label"] == "REAL" else "red"
    fig.suptitle(f"Prediction: {result['label']} (confidence {result['confidence'] * 100:.2f}%)", color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, nargs="+")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--panel-dir", type=Path, default=Path("explanations"))
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    args.panel_dir.mkdir(parents=True, exist_ok=True)
    for path in args.image:
        result = predict(model, path, tta=not args.no_tta)
        panel = args.panel_dir / f"{Path(path).stem}_explained.png"
        save_panel(result, panel)
        print(json.dumps({"image": path, **{k: v for k, v in result.items() if k != "bgr"}, "panel": str(panel)}))


if __name__ == "__main__":
    main()
