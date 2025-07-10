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
from meeting_analyzer import analyze_meeting


def run_pipeline(audio_file: str, 
                 output_dir: str = None, 
                 formats: list = None,
                 confidence_threshold: float = None,
                 chunk_size: int = None,
                 language: str = None,
                 run_llm_analysis: bool = True) -> dict:
    """
    Run the complete conversation analysis pipeline.
    
    Args:
        audio_file: Path to audio or video file
        output_dir: Directory for outputs (default: config.OUTPUT_DIR)
        formats: Output formats list (default: config.DEFAULT_OUTPUT_FORMATS)
        confidence_threshold: Min confidence for speaker assignment
        chunk_size: Chunk size in seconds
        language: Language code (auto/en/ar)
        run_llm_analysis: Whether to run LLM meeting analysis
    
    Returns:
        dict: Pipeline results with success status, file paths, and metadata
    """
    # Set defaults
    output_dir = output_dir or config.OUTPUT_DIR
    formats = formats or config.DEFAULT_OUTPUT_FORMATS
    confidence_threshold = confidence_threshold or config.DEFAULT_CONFIDENCE_THRESHOLD
    chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
    language = language or config.DEFAULT_LANGUAGE
    
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        "success": False,
        "error": None,
        "outputs": {},
        "metadata": {},
        "stats": {}
    }
    
    try:
        # --- Step 0: Audio Extraction (if needed) ---
        print("\n[0/4] Checking input file and extracting audio if needed...")
        audio_for_processing = extract_audio_if_needed(audio_file)
        temp_audio_created = audio_for_processing != audio_file

        try:
            # --- Step 1: Transcription ---
            print("\n[1/4] Running Whisper transcription...")
            transcriber = OptimizedParallelTranscriber(model_size="large-v3")
            transcribe_result = transcriber.transcribe_file(
                audio_for_processing,
                language=language,
                output_formats=[]  # We'll handle output here
            )
            if not transcribe_result.get("success"):
                result["error"] = f"Transcription failed: {transcribe_result.get('error')}"
                return result
            transcription_segments = transcribe_result["segments"]

            # --- Step 2: Diarization ---
            print("\n[2/4] Running speaker diarization...")
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
        print("\n[3/4] Aligning transcription and diarization...")
        analyzer = ConversationAnalyzer(confidence_threshold=confidence_threshold)
        aligned = analyzer.analyze(transcription_segments, diarization_segments)

        # --- Step 4: Generate outputs ---
        print("\n[4/4] Generating outputs...")
        metadata = {
            'audio_file': audio_file,
            'pipeline_version': config.PIPELINE_VERSION,
            'confidence_threshold': confidence_threshold,
            'language': language,
            'total_segments': len(aligned)
        }
        
        base = os.path.splitext(os.path.basename(audio_file))[0]
        output_paths = {}

        if 'html' in formats:
            html_path = os.path.join(output_dir, f'{base}.html')
            render_html(aligned, html_path, metadata)
            output_paths['html'] = html_path
            
        if 'json' in formats:
            json_path = os.path.join(output_dir, f'{base}.json')
            render_json(aligned, json_path, metadata)
            output_paths['json'] = json_path
            
            # Automatically run LLM meeting analysis on the generated JSON
            if run_llm_analysis:
                try:
                    print("\n[5/5] Running LLM meeting analysis (OpenAI GPT-4o)...")
                    analyze_meeting(json_path)
                    # Add meeting summary paths
                    output_paths['meeting_summary_txt'] = json_path.replace('.json', '.meeting_summary.txt')
                    output_paths['meeting_summary_html'] = json_path.replace('.json', '.meeting_summary.html')
                except Exception as e:
                    print(f"⚠️ LLM meeting analysis failed: {e}")
                    result["llm_analysis_error"] = str(e)
                    
        if 'srt' in formats:
            srt_path = os.path.join(output_dir, f'{base}.srt')
            render_srt(aligned, srt_path)
            output_paths['srt'] = srt_path

        # Calculate stats
        from collections import defaultdict
        speaker_stats = defaultdict(float)
        total_duration = 0
        for seg in aligned:
            duration = seg.get('end', 0) - seg.get('start', 0)
            speaker_stats[seg.get('speaker', 'Unknown')] += duration
            total_duration += duration
        
        stats = {
            'total_duration': total_duration,
            'num_speakers': len(speaker_stats),
            'num_segments': len(aligned),
            'speaker_durations': dict(speaker_stats)
        }

        result.update({
            "success": True,
            "outputs": output_paths,
            "metadata": metadata,
            "stats": stats,
            "aligned_segments": aligned  # Include for API responses
        })
        
        print(f"✅ Pipeline completed successfully! Outputs written to {output_dir}")
        return result
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Pipeline failed: {e}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Speaker-aware multilingual transcription pipeline")
    parser.add_argument('audio_file', help='Path to audio or video file')
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR, help='Directory for outputs')
    parser.add_argument('--formats', default=','.join(config.DEFAULT_OUTPUT_FORMATS), help='Output formats (comma-separated: html,json,srt)')
    parser.add_argument('--confidence-threshold', type=float, default=config.DEFAULT_CONFIDENCE_THRESHOLD, help='Min confidence for speaker assignment')
    parser.add_argument('--chunk-size', type=int, default=config.DEFAULT_CHUNK_SIZE, help='Chunk size in seconds')
    parser.add_argument('--language', default=config.DEFAULT_LANGUAGE, help='Language (auto/en/ar)')
    args = parser.parse_args()

    # Convert formats string to list
    formats = [f.strip() for f in args.formats.split(',')]
    
    # Run the pipeline
    result = run_pipeline(
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        formats=formats,
        confidence_threshold=args.confidence_threshold,
        chunk_size=args.chunk_size,
        language=args.language
    )
    
    if not result["success"]:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
