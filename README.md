Audio-Sentinel V2: High-Fidelity Synthetic Voice Detection Framework
An end-to-end Deep Learning pipeline for distinguishing authentic human speech from AI-generated audio using Transformer-based architectures.

Project Vision & Evolution
The rapid proliferation of synthetic voice technology (estimated to reach a $15B market by 2030) has created a critical need for robust forensic tools. This project documents an iterative engineering process to build a scalable classifier capable of detecting deepfakes.

Instead of a static model, this repository showcases a multi-stage development lifecycle:

V1 (Proof of Concept): Validated Wav2Vec2 on 100-200 local samples.

V2 (Infrastructure Pivot): Transitioned to a Streaming Architecture to handle 4,000+ samples without local disk bottlenecks.

V3 (Advanced Interleaving): Implemented real-time dataset balancing and normalization for diversified AI-audio profiles.

Technical Methodology & Pipeline
1. Data Orchestration (The Streaming Solution)
To overcome the Disk Space Constraint identified in early iterations (Slide 20), the framework utilizes Hugging Face Dataset Streaming:

Source A (Real): mozilla-foundation/common_voice_11_0 (2,000 samples).

Source B (AI): andi611/wavefake-audio (2,000 samples).

Logic: Utilized interleave_datasets with a 50/50 probability split and a 500-sample shuffle buffer to ensure stochastic gradient descent stability while maintaining a zero-disk footprint.

2. Feature Engineering & Signal Processing
The pipeline implements a specialized preprocessing wrapper using pydub and torchaudio:

Resampling: All audio is strictly downsampled to 16kHz to match the Wav2Vec2 pre-training manifold.

Normalization: Applied audio amplitude normalization to mitigate volume bias between datasets.

Feature Extraction: Leverages the Wav2Vec2FeatureExtractor to convert raw waveforms into high-dimensional latent representations.

3. Model Architecture
Backbone: Wav2Vec2ForSequenceClassification (Base-960h).

Fine-Tuning: The transformer layers were fine-tuned for binary classification (Real vs. AI).

Optimization: Configured for high-throughput inference, capable of processing any standard audio format through an automated resampling wrapper.

Challenges & Engineering Bottlenecks
This project served as a rigorous study in Resource-Constrained Deep Learning:

Computational VRAM Limits: Fine-tuning 95M+ parameter models on T4 GPUs required optimizing batch sizes and sequence lengths to prevent Out-of-Memory (OOM) errors.

The "Deepfake" Generalization Gap: While the model achieved high accuracy on standard AI-generated audio, we identified a performance drop-off when encountering sophisticated Deepfakes (Slide 19).

Data Scarcity: Access to high-quality, diverse synthetic audio profiles remains a primary bottleneck for global generalization.

Key Learnings & Roadmap
Iteration Success: Successfully scaled from 100 to 4,000 training samples while maintaining a clean memory profile.

Framework Maturity: Developed a modular inference engine that supports real-time classification of external audio files.

Future Vision: Transitioning to a Multi-Classification Model (Real vs. AI vs. Deepfake) and utilizing higher-tier A100/H100 clusters to incorporate larger, more diverse datasets (Slide 21).

Tech Stack
Deep Learning: PyTorch, Hugging Face Transformers

Data Handling: Hugging Face datasets (Streaming Mode), interleave_datasets

Audio Processing: torchaudio, pydub, librosa

Optimization: Wav2Vec2 Feature Extraction

Project Structure
core/data_pipeline.py: Streaming and interleaving logic.

core/model_engine.py: Model definition and feature extraction.

notebooks/AI_Audio_Classifier.ipynb: The original research and training log.

EVOLUTION.md: Detailed breakdown of iterative improvements and pivots.
