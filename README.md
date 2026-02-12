[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoshanNair1214/audio-sentinel-v2-framework/blob/main/notebooks/AI_Audio_Classifier.ipynb)
# Audio-Sentinel V2: High-Fidelity Synthetic Voice Detection Framework

**An advanced Deep Learning infrastructure for distinguishing authentic human speech from AI-generated audio and biometric deepfakes.**

---

## Project Vision & Engineering Narrative
As the synthetic audio market accelerates toward a projected **$15B valuation by 2030**, the boundary between organic and generated speech has blurred. This project documents an aggressive engineering journey to build a forensic-grade classifier. 

Rather than a static script, this repository showcases a **multi-stage architectural evolution**—solving critical bottlenecks in data streaming, GPU memory management (OOM), and spectral feature leakage.

---

## Technical Architecture & Methodology



### 1. Cloud-Native Streaming Infrastructure
To overcome the **Local Disk Exhaustion** identified in Phase 2 (Slide 20), I pivoted from traditional data loading to a **Hugging Face Streaming Pipeline**:
* **Data Interleaving:** Orchestrated the blending of **4,000 high-fidelity samples** using `interleave_datasets`.
    * **Authentic Domain:** `mozilla-foundation/common_voice_11_0` (English subset).
    * **Synthetic Domain:** `andi611/wavefake-audio` (Engineered AI vocoder profiles).
* **Buffer Management:** Implemented a **500-sample shuffle buffer** to maintain stochasticity in the training stream without requiring local storage.
* **Authentication:** Integrated Hugging Face `User Access Tokens` for secure, API-led data ingestion.

### 2. Signal Processing & Feature Engineering
The framework treats raw audio as a digital signal that must be aligned with the transformer's latent manifold:
* **Dynamic Resampling:** Utilized `torchaudio` and `pydub` to downsample all inputs to a strict **16,000 Hz**, matching the `Wav2Vec2` pre-training frequency.
* **Feature Extraction:** Leveraged `Wav2Vec2FeatureExtractor` to transform time-series waveforms into normalized tensors, effectively mitigating volume bias and background noise artifacts.



### 3. Model Engine: Wav2Vec2 Pipeline
The core system is a fine-tuned **Transformer-based Classifier**:
* **Backbone:** `Wav2Vec2ForSequenceClassification` (95M+ Parameters).
* **Optimization Strategy:** The model was fine-tuned for binary classification, learning to isolate the specific "spectral signatures" left by neural vocoders versus the natural jitter of human vocal folds.
* **Inference Wrapper:** Built a versatile inference engine supporting `.mp3`, `.wav`, and `.m4a`, performing real-time normalization before classification.

---

## Engineering Challenges & Iterative Improvements

### **Iteration 1: Baseline Validation**
* **Goal:** Verify sensitivity to synthetic artifacts.
* **Outcome:** Successfully trained on 100-200 local samples; identified that standard normalization was insufficient for cross-dataset generalization.

### **Iteration 2: The Scaling Pivot**
* **Challenge:** Encountered GPU VRAM OOM errors and local storage limits when scaling to 4,000 samples.
* **Solution:** Developed the **Streaming Framework** and implemented **Gradient Accumulation** to simulate larger batch sizes on a single T4 GPU instance.

### **Iteration 3: Forensic Gap Analysis**
* **Discovery:** While binary classification reached near-perfect accuracy on standard TTS, the model showed sensitivity to high-complexity **Biometric Deepfakes**.
* **Roadmap:** Proposed a transition to **Multi-Class Classification** (Real | AI | Deepfake) to isolate synthesis artifacts from mimicry (Slide 21).

---

## Tech Stack
* **Frameworks:** PyTorch, Hugging Face Transformers, Hugging Face Datasets
* **Audio Engineering:** Pydub, Torchaudio, Librosa
* **Models:** Facebook/Wav2Vec2-Base-960h
* **Optimization:** ASHA-informed resource management, Data Streaming (HF Tokens)

---

## Performance Summary
| Metric | Baseline (Local) | Framework V2 (Streamed) |
| :--- | :--- | :--- |
| **Dataset Size** | 200 Samples | **4,000 Samples** |
| **Storage Strategy** | Local Disk (5GB+) | **HF Streaming (~0GB)** |
| **Accuracy (AI/Real)**| 100.0% | **99.9% (Validated)** |
| **Processing Complexity**| $O(N)$ Disk | **$O(1)$ Disk Growth** |

---

## Project Structure
* **`core/data_pipeline.py`**: Interleaving logic and streaming wrappers.
* **`core/model_engine.py`**: Model initialization and inference logic.
* **`notebooks/`**: Research logs including `AI_Audio_Classifier.ipynb`.
* **`EVOLUTION.md`**: Detailed technical breakdown of pivots and hardware workarounds.

---

### Conclusion
The **Audio-Sentinel V2** framework is a demonstration of **Engineering Resilience**. By moving to a streaming-first architecture, I proved that high-parameter Transformer models can be successfully trained and optimized under significant hardware and storage constraints.
