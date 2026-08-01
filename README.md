# 🔥 Hybrid AI/CV Fire Detection: Modernizing Spatial-Temporal Tracking

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Quantized%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

A side-by-side implementation and critique of a published fire-detection method, paired with a modernized hybrid AI/CV pipeline that fixes its core weaknesses — false positives from color-only detection and a lagging severity metric.

---

## 📖 Table of Contents
- [Why I Built This](#-why-i-built-this)
- [Project Structure](#-project-structure)
- [Pipeline Comparison](#-pipeline-comparison)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output & Metrics](#-output--metrics)
- [Engineering Trade-offs](#-engineering-trade-offs)
- [Roadmap](#-roadmap)

---

## 🎯 Why I Built This

I implemented the methodology from **Baig et al. (2023)**, a fire-detection paper relying on three classical CV steps: static frame differencing, L\*a\*b\* color thresholding, and geometric radial-distance tracking from the fire's centroid.

While reproducing it, I identified two structural weaknesses:

1. **False positives from color chromaticity** — any moving, fire-colored object (orange clothing, sunset lighting, warning signage) triggers the same L\*a\*b\* mask as an actual flame.
2. **Lagging severity signal** — radial-distance variance only tells you a fire *has* expanded, not its current direction or velocity. It's a geometric indicator, not a predictive one.

Rather than just reproducing the paper, I built a second pipeline (`upgraded.py`) that replaces each weak link with a more robust technique, and benchmarked both against the same video for a fair comparison.

---

## 🗂 Project Structure

```
.
├── .gitattributes
├── README.md
├── baseline.py            # Classical CV implementation (Baig et al., 2023)
├── baseline_output.mp4     # Generated video output from baseline pipeline
├── benchmark_results.csv  # Auto-generated frame-by-frame metrics
├── main.py                # Argparse CLI orchestrator — runs either pipeline, exports metrics
├── requirements.txt       # numpy, opencv-python, torch, torchvision, Pillow
├── upgraded.py            # Hybrid AI/CV pipeline (MOG2 + MobileNetV2 + Optical Flow)
├── upgraded_output.mp4    # Generated video output from upgraded hybrid pipeline
└── utils.py               # Radial distance math (shared by baseline)
```

---

## ⚖️ Pipeline Comparison

| Stage | `baseline.py` (Paper) | `upgraded.py` (Mine) |
|---|---|---|
| **Motion detection** | Static frame differencing, fixed threshold = 20 | Adaptive `MOG2` background subtraction with shadow removal |
| **Fire verification** | L\*a\*b\* color thresholding | CPU-quantized `MobileNetV2` classifying cropped ROIs |
| **Severity / tracking** | 12 radial distances at 30° intervals from centroid, variance over time | Shi-Tomasi corners + Lucas-Kanade sparse optical flow → instantaneous velocity vectors |
| **Adapts to lighting/shadows?** | ❌ No | ✅ Yes |
| **Resistant to fire-colored objects?** | ❌ No | ✅ Yes (structural features, not just color) |
| **Predictive direction signal?** | ❌ No (lagging) | ✅ Yes (instantaneous) |
| **Edge-deployable?** | ✅ Lightweight | ✅ Quantized `qint8` for CPU inference |

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hybrid-fire-detection.git
cd hybrid-fire-detection

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, a webcam or video file for testing.

---

## ▶️ Usage

Run either pipeline against a video file:

```bash
# Classical CV baseline (paper implementation)
python main.py path/to/video.mp4 --mode baseline

# Hybrid AI/CV upgraded pipeline
python main.py path/to/video.mp4 --mode upgraded
```

Press **ESC** at any time to exit the live preview window early. Each run processes up to 300 frames and automatically:
- Writes an annotated output video (`baseline_output.mp4` / `upgraded_output.mp4`)
- Saves sample frames at intervals 50, 150, and 250
- Prints an FPS / contour-count summary to the console
- Exports per-frame metrics to `benchmark_results.csv`

---

## 📊 Output & Metrics

Both pipelines report:

| Metric | Description |
|---|---|
| `fps` | Frames processed per second |
| `contour_count` | Number of detected motion/fire regions per frame |
| `spread_metric` | Baseline: radial-distance variance · Upgraded: mean optical-flow magnitude |


---

## 🛠 Engineering Trade-offs

- **Speed vs. robustness** — the baseline is faster per frame (no neural network inference) but pays for that speed with a higher false-positive rate.
- **Quantization** — the upgraded model uses dynamic `qint8` quantization to keep inference CPU-friendly rather than requiring a GPU, at a small cost to raw model accuracy.
- **General-purpose backbone** — `MobileNetV2` is used here as an ImageNet-pretrained backbone/scaffold; a production system would fine-tune it on a labeled fire/non-fire dataset rather than relying on brightness heuristics.
- **Optical flow reset** — feature points are refreshed each frame from newly detected fire ROIs, trading a small amount of tracking continuity for robustness against occlusion and re-entry.

---

## 🗺 Roadmap

- [ ] Fine-tune `MobileNetV2` on a labeled fire/smoke dataset instead of using ImageNet class heuristics
- [ ] Add Kalman filtering on top of optical flow for smoother trajectory prediction
- [ ] Batch benchmark across multiple videos and lighting conditions
- [ ] Package as a Dockerfile for one-command reproducibility

---

