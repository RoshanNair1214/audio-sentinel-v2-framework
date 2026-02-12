import argparse
from core.data_pipeline import AudioForensicStreamer
from core.model_engine import AudioSentinelEngine

def run_streaming_demo(token):
    """
    Demonstrates the streaming and interleaving logic.
    """
    print("\n--- Phase 2: Streaming Data Pipeline ---")
    streamer = AudioForensicStreamer(hf_token=token)
    pipeline = streamer.get_pipeline(real_limit=5, ai_limit=5)
    
    for i, sample in enumerate(pipeline):
        print(f"Sample {i+1}: Length {len(sample['audio']['array'])} | Rate {sample['audio']['sampling_rate']}")

def run_inference(file_path):
    """
    Runs forensic classification on a single local file.
    """
    print("\n--- Phase 3: Forensic Inference Engine ---")
    engine = AudioSentinelEngine()
    result = engine.classify_audio(file_path)
    
    print(f"File: {file_path}")
    print(f"Result: {result['label']}")
    print(f"Confidence: {result['confidence']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-Sentinel V2 Framework")
    parser.add_argument("--mode", type=str, choices=["stream", "predict"], required=True,
                        help="Choose 'stream' to test data pipeline or 'predict' for inference.")
    parser.add_argument("--file", type=str, help="Path to audio file for prediction.")
    parser.add_argument("--token", type=str, help="Hugging Face User Access Token (required for streaming).")

    args = parser.parse_args()

    if args.mode == "stream":
        if not args.token:
            print("❌ Error: Streaming requires a --token.")
        else:
            run_streaming_demo(args.token)
    
    elif args.mode == "predict":
        if not args.file:
            print("❌ Error: Prediction requires a --file path.")
        else:
            run_inference(args.file)
