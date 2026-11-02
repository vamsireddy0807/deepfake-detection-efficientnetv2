# Deepfake Face Detection: EfficientNetV2-B0 with OpenCV Forensics

A real-vs-fake face classifier trained on 140,000 images. It reaches **96.25% accuracy and 0.9972 ROC-AUC on a held-out 20,000-image test set**, with OpenCV forensic panels (ELA, DCT, FFT, Laplacian, Sobel) that show *why* an image looks synthetic. Training takes about **32 minutes on a single Colab T4**.

## Results

All numbers come from the run saved in [`notebooks/deepfake_detection.ipynb`](notebooks/deepfake_detection.ipynb). The test split (10,000 real and 10,000 fake) is never used for training or checkpoint selection.

| Test set (20,000 images) | Accuracy | ROC-AUC |
|---|---|---|
| Single pass | 96.07% | 0.9967 |
| **Horizontal-flip TTA (x2)** | **96.25%** | **0.9972** |

| Class (TTA) | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| fake | 0.9365 | 0.9922 | 0.9635 | 10,000 |
| real | 0.9917 | 0.9327 | 0.9613 | 10,000 |

Best validation AUC was 0.9953 (20,000 validation images). The model misses very few fakes (99.2% recall on `fake`) and errs towards flagging real faces, which is the safer failure mode for screening.

## How it works

**Data.** [`xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) provides 100K training, 20K validation and 20K test images, balanced 50/50. Real faces come from FFHQ; fakes are StyleGAN-generated. The faces are already cropped, so the pipeline skips video frame extraction, which is usually the slowest step in deepfake pipelines.

**Model.**
* ImageNet EfficientNetV2-B0 backbone with built-in rescaling, followed by global average pooling, BatchNorm, Dropout 0.3, Dense 128, Dropout 0.25 and a sigmoid output. The output layer stays in float32 for numerical stability.
* Input size 160x160, batch size 128, `mixed_float16` precision, and a `tf.data` pipeline with parallel decoding and prefetching so the GPU never waits on I/O.

**Training (10 epochs, about 31.5 minutes on a T4).**
1. *Head only:* 3 epochs with the backbone frozen, Adam at 1e-3. Validation AUC reaches 0.945.
2. *Fine-tune:* the top 30% of backbone layers are unfrozen and trained for 7 epochs, Adam at 1e-4 with 1-epoch linear warm-up and cosine decay. Checkpoints are kept on best validation AUC, with early stopping on validation loss (patience 3). Validation AUC rises from 0.970 to 0.995.

**Augmentation.** Horizontal flip and tight brightness, contrast and saturation jitter. Heavy distortion destroys the subtle artefacts the model relies on.

**Explanations.** `deepfake_detection.forensics` implements five OpenCV operators:
* *Error Level Analysis:* GAN faces have uniformly low JPEG-recompression residuals.
* *DCT and FFT log-spectra:* StyleGAN's transposed convolutions leave grid patterns at high frequencies.
* *Laplacian and Sobel residuals:* high-pass filters reveal synthesis seams.

`scripts/predict.py` saves these panels next to each prediction. They are for interpretation; the classifier itself sees RGB pixels only.

## Quick start

```bash
pip install -r requirements.txt
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data --unzip
python scripts/train.py --root data/real_vs_fake/real-vs-fake --out artifacts/
python scripts/evaluate.py --root data/real_vs_fake/real-vs-fake \
    --model artifacts/effv2b0_deepfake_best.keras --out artifacts/test_metrics.json
python scripts/predict.py --model artifacts/effv2b0_deepfake_best.keras --image face1.jpg face2.jpg
pytest                                  # forensic operator tests (OpenCV + NumPy only)
```

To run in Colab, open the notebook, choose a GPU runtime, upload `kaggle.json` when prompted, and run all cells. It takes about 35 minutes end to end.

## Repository layout

```
notebooks/deepfake_detection.ipynb    end-to-end run with saved outputs (the run of record)
src/deepfake_detection/forensics.py   ELA, DCT, FFT, Laplacian, Sobel operators
src/deepfake_detection/data.py        dataset indexing and tf.data pipeline
src/deepfake_detection/model.py       EfficientNetV2-B0 model, layer unfreezing, warm-up cosine schedule
scripts/train.py                      two-phase training -> best checkpoint and history JSON
scripts/evaluate.py                   test metrics, single pass and TTA -> JSON
scripts/predict.py                    REAL/FAKE prediction with saved forensic panels
tests/test_forensics.py               unit tests for the forensic operators
```

## Limitations

* The fakes come from a single generator family (StyleGAN, on FFHQ-style aligned faces). Performance on diffusion images, face swaps, re-enactment and compressed social-media video has **not** been evaluated and will be lower without further training data.
* The input is aligned face crops. For full frames or video, run a face detector first, then aggregate per-frame scores.
* A 0.5 threshold is used throughout; calibrate it on your own data for operational use.

## License

MIT for this code. The dataset keeps its Kaggle license, and the EfficientNetV2 ImageNet weights come from Keras Applications.
