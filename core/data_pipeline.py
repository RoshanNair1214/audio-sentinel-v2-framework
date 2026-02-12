import os
from datasets import load_dataset, interleave_datasets, Audio

class AudioForensicStreamer:
    """
    Orchestrates high-scale audio data ingestion using Hugging Face Streaming.
    Solves local storage bottlenecks by interleaving real and synthetic datasets.
    """
    def __init__(self, hf_token: str = None):
        self.token = hf_token

    def get_pipeline(self, real_limit=2000, ai_limit=2000):
        print(f"📡 Initializing Stream: {real_limit} Real | {ai_limit} AI samples")
        
        # 1. Stream Real Audio (Mozilla Common Voice)
        # Note: 'streaming=True' ensures zero-disk footprint
        real_ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", 
                               split="train", streaming=True, token=self.token)
        real_ds = real_ds.take(real_limit)

        # 2. Stream AI Audio (Wavefake)
        ai_ds = load_dataset("andi611/wavefake-audio", split="train", streaming=True)
        ai_ds = ai_ds.take(ai_limit)

        # 3. Interleave and Shuffle
        # We use a 500-sample buffer to ensure stochasticity in the stream
        dataset = interleave_datasets([real_ds, ai_ds], probabilities=[0.5, 0.5], seed=42)
        
        # 4. Cast to 16kHz (The Wav2Vec2 requirement)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset.shuffle(buffer_size=500)
