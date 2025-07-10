"""
conversation_transcriber.py - Main CLI for the conversation analysis pipeline

- Accepts audio/video files and config options
- Runs transcription, diarization, alignment, and rendering
- Provides progress and error handling
"""


import argparse
import os
from utils import ConversationError, extract_audio_if_needed
import config


# --- Real pipeline imports ---
from final_optimized_transcriber import OptimizedParallelTranscriber
from speaker_diarizer import get_hf_token
from conversation_analyzer import ConversationAnalyzer
from conversation_renderer import render_html, render_json, render_srt


def main():
    parser = argparse.ArgumentParser(description="Speaker-aware multilingual transcription pipeline")
    parser.add_argument('audio_file', help='Path to audio or video file')
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR, help='Directory for outputs')
    parser.add_argument('--formats', default=','.join(config.DEFAULT_OUTPUT_FORMATS), help='Output formats (comma-separated: html,json,srt)')
    parser.add_argument('--confidence-threshold', type=float, default=config.DEFAULT_CONFIDENCE_THRESHOLD, help='Min confidence for speaker assignment')
    parser.add_argument('--chunk-size', type=int, default=config.DEFAULT_CHUNK_SIZE, help='Chunk size in seconds')
    parser.add_argument('--language', default=config.DEFAULT_LANGUAGE, help='Language (auto/en/ar)')
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 0: Audio Extraction (if needed) ---
    print("\n[0/3] Checking input file and extracting audio if needed...")
    audio_for_processing = extract_audio_if_needed(args.audio_file)
    temp_audio_created = audio_for_processing != args.audio_file

    try:
        # --- Step 1: Transcription ---
        print("\n[1/3] Running Whisper transcription...")
        transcriber = OptimizedParallelTranscriber(model_size="large-v3")
        transcribe_result = transcriber.transcribe_file(
            audio_for_processing,
            language=args.language,
            output_formats=[]  # We'll handle output here
        )
        if not transcribe_result.get("success"):
            print(f"Transcription failed: {transcribe_result.get('error')}")
            return
        transcription_segments = transcribe_result["segments"]

        # --- Step 2: Diarization ---
        print("\n[2/3] Running speaker diarization...")
        from pyannote.audio import Pipeline
        hf_token = get_hf_token()
        pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL_NAME,
            use_auth_token=hf_token
        )
        diarization = pipeline(audio_for_processing)
        diarization_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": speaker
            })

    finally:
        # Clean up temp audio file if created
        if temp_audio_created:
            try:
                os.remove(audio_for_processing)
                print(f"🧹 Temp audio file removed: {audio_for_processing}")
            except Exception as e:
                print(f"⚠️ Could not remove temp audio file: {e}")


    # --- Step 3: Alignment ---
    analyzer = ConversationAnalyzer(confidence_threshold=args.confidence_threshold)
    try:
        aligned = analyzer.analyze(transcription_segments, diarization_segments)
    except ConversationError as e:
        print(f"Error: {e}")
        return

    metadata = {
        'audio_file': args.audio_file,
        'pipeline_version': config.PIPELINE_VERSION,
        'confidence_threshold': args.confidence_threshold
    }
    formats = [f.strip() for f in args.formats.split(',')]
    base = os.path.splitext(os.path.basename(args.audio_file))[0]
    if 'html' in formats:
        render_html(aligned, os.path.join(args.output_dir, f'{base}.html'), metadata)
    if 'json' in formats:
        render_json(aligned, os.path.join(args.output_dir, f'{base}.json'), metadata)
    if 'srt' in formats:
        render_srt(aligned, os.path.join(args.output_dir, f'{base}.srt'))
    print(f"Outputs written to {args.output_dir}")

if __name__ == "__main__":
    main()
