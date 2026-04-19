# AMP 2026 - Team J: Onset & Beat Tracking Project

## 1. Project Overview
This repository contains the implementation for the Onset Detection, Beat Tracking, and Tempo Estimation components.

**Current Status:** Onset Detection (CNN) is finalized with a Cross-Validation F1-score of **0.809**.

---

## 2. Technical Specifications (Synchronization)
To ensure all modules align, please use the following parameters for all audio processing:

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Sample Rate** | 44100 Hz | Always downsample/upsample to this |
| **Hop Length** | 441 samples | **10ms resolution** - CRITICAL for alignment |
| **Window Size** | 1024 samples | ~23ms (Hanning Window) |
| **Mel Bands** | 80 | Log-compressed Mel Spectrogram |

---

## 3. Onset Detection (Diana)
We moved from a Spectral Flux baseline to a **3-Layer CNN** to improve robustness against noise and complex textures.

### Key Features:
* **Architecture:** 3 Conv Layers + Batch Normalization + Dropout (0.3).
* **Context:** 31-frame sliding window (~310ms of temporal context).
* **Label Widening:** Targets were widened to 3 frames during training to improve convergence.
* **Output:** Smoothed probability curve (0.0 to 1.0).

### How to use:
1. **Inference:** Run `python src/format_json.py` to generate the server submission.
2. **Activations:** Run `python src/export_activations.py` to generate `.npy` files for the beat tracker.

---

## 4. Roadmap & Team Tasks

### ✅ Onset Detection (Completed)
- [x] Implement Log-Mel Spectrogram features.
- [x] Train CNN model (`onset_cnn_v1.pth`).
- [x] Optimize peak picking (Best Params: `delta=0.15`, `wait=3`).
- [x] Export onset activations for Teammate 2.

### ⏳ Beat Tracking (Pending)
- [ ] Load `_act.npy` files from `data/activations/`.
- [ ] Implement Dynamic Programming (DP) or HMM-based beat tracking.
- [ ] Sync beat periods with the 10ms hop length.
- [ ] Evaluate against ground truth using `mir_eval.beat`.

### ⏳ Tempo Estimation (Pending)
- [ ] Extract global tempo (BPM) from the periodicity of the onset activation.
- [ ] Implement ACL (Autocorrelation) or Comb Filter bank.
- [ ] Handle "Double/Half Tempo" errors.

---

## 5. Setup & Installation
```bash
# Create and activate environment
python -m venv .venv
source .venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
