import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

class AudioSentinelEngine:
    """
    The Core Inference Engine utilizing Wav2Vec2 for Sequence Classification.
    Handles dynamic resampling and forensic scoring.
    """
    def __init__(self, model_id="facebook/wav2vec2-base-960h", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id, num_labels=2).to(self.device)

    def classify_audio(self, file_path: str):
        """
        Performs inference on a raw audio file with automated resampling.
        """
        # Load and Resample to 16kHz
        speech, sr = torchaudio.load(file_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech)

        # Preprocessing via Feature Extractor
        inputs = self.extractor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forensic Scoring
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        prediction = torch.argmax(logits, dim=-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()

        label_map = {0: "REAL (Human)", 1: "AI (Generated)"}
        return {"label": label_map[prediction], "confidence": f"{confidence*100:.2f}%"}
