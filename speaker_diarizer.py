#!/usr/bin/env python3
"""
Speaker Diarization Module - Step 2: HuggingFace Authentication
Adds HF token support and tests actual diarization on sample audio
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

print("🎭 Speaker Diarizer - Step 2: HuggingFace Authentication")

def get_hf_token():
    """Get HuggingFace token from environment or user input"""
    print("🔑 Checking for HuggingFace authentication...")
    
    # Check environment variable first
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    config_path = Path.home() / '.hf_token'

    # Check config file if not in environment
    if not token and config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            token = f.read().strip()
        if token:
            print(f"✅ Found HF token in {config_path}")
            return token

    if token:
        print("✅ Found HF token in environment variable")
        return token

    # If no token in environment or config, provide instructions
    print("ℹ️  No HuggingFace token found in environment or config file.")
    print("📝 To use speaker diarization, you need to:")
    print("   1. Visit https://hf.co/pyannote/speaker-diarization-3.1")
    print("   2. Accept the user conditions")
    print("   3. Create a token at https://hf.co/settings/tokens")
    print("   4. Set environment variable: set HF_TOKEN=your_token_here")
    print("   5. Or save your token in the file: ~/.hf_token")
    print()

    # Ask user if they want to input token now
    response = input("Would you like to input your HF token now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        token = input("Enter your HuggingFace token: ").strip()
        if token:
            # Ask to save
            save = input(f"Save this token to {config_path} for future use? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(token)
                    print(f"💾 Token saved to {config_path}")
                except Exception as e:
                    print(f"⚠️  Could not save token: {e}")
            return token

    print("⚠️  Continuing without token (limited functionality)")
    return None

def test_imports():
    """Test if all required imports work correctly"""
    print("📦 Testing imports...")
    
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote.audio imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pyannote.audio: {e}")
        return False

def test_pipeline_loading(hf_token=None):
    """Test if we can load the diarization pipeline with authentication"""
    print("🔄 Testing pipeline loading...")
    
    try:
        from pyannote.audio import Pipeline
        
        # Try to load the pipeline with authentication
        if hf_token:
            try:
                print("🔐 Attempting to load pipeline with HF token...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                print("✅ Speaker diarization pipeline loaded successfully with authentication!")
                return True, pipeline
            except Exception as auth_error:
                print(f"❌ Authentication failed: {auth_error}")
                return False, None
        else:
            print("⚠️  No HF token provided - pipeline loading will be skipped")
            return True, None
            
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return False, None

def find_sample_audio():
    """Prompt user to select an audio file for testing"""
    print("🎵 Please enter the path to the audio file you want to test:")
    audio_file = input("Audio file path: ").strip()
    if not audio_file or not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return None
    print(f"✅ Selected audio file: {audio_file}")
    return audio_file

def test_basic_diarization(pipeline, audio_file):
    """Test basic diarization functionality"""
    if not pipeline or not audio_file:
        print("⚠️  Skipping diarization test (missing pipeline or audio)")
        return True
    
    print(f"🎯 Testing diarization on: {audio_file}")
    
    import csv
    import json
    from datetime import datetime
    log_file = f"diarization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # For unfiltered diarization
    unfiltered_segments = []
    try:
        # Run diarization (this will take some time)
        print("⏳ Running speaker diarization (this may take a moment)...")
        start_time = time.time()
        diarization = pipeline(str(audio_file))
        end_time = time.time()
        print(f"✅ Diarization completed in {end_time - start_time:.2f} seconds")

        # Collect speaker labels and total duration, remove duplicates
        speakers = set()
        total_duration = 0.0
        segments = []
        seen = set()
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # Save unfiltered
            unfiltered_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker
            })
            # Filtered
            key = (round(segment.start, 2), round(segment.end, 2), speaker)
            if key in seen:
                continue
            seen.add(key)
            speakers.add(speaker)
            duration = segment.end - segment.start
            total_duration += duration
            segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": speaker
            })

        print(f"📊 Results: {len(speakers)} speakers detected, {total_duration:.2f}s total speech")

        # Show first few segments
        print("🎤 First few speaker segments:")
        for i, seg in enumerate(segments):
            if i >= 3:
                break
            print(f"   {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['speaker']}")

        # Ask user if they want to see all segments
        show_all = input("\nShow all diarization segments? (y/n): ").strip().lower()
        if show_all in ["y", "yes"]:
            print("\nAll diarization segments:")
            for seg in segments:
                print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['speaker']}")

        # Ask user to save results
        save = input("\nSave diarization results to file? (y/n): ").strip().lower()
        if save in ["y", "yes"]:
            base = Path(audio_file).stem
            # Save as CSV (filtered)
            csv_path = f"{base}_diarization.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["start", "end", "speaker"])
                writer.writeheader()
                writer.writerows(segments)
            print(f"✅ CSV saved: {csv_path}")
            # Save as JSON (filtered)
            json_path = f"{base}_diarization.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)
            print(f"✅ JSON saved: {json_path}")
            # Save as TXT (filtered)
            txt_path = f"{base}_diarization.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['speaker']}\n")
            print(f"✅ TXT saved: {txt_path}")
            # Save unfiltered JSON
            unfiltered_json_path = f"{base}_diarization_unfiltered.json"
            with open(unfiltered_json_path, 'w', encoding='utf-8') as f:
                json.dump(unfiltered_segments, f, indent=2)
            print(f"✅ Unfiltered JSON saved: {unfiltered_json_path}")

        # Log output
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write(f"Diarization run for: {audio_file}\n")
            log.write(f"Speakers: {len(speakers)}\n")
            log.write(f"Total speech: {total_duration:.2f}s\n")
            log.write(f"Segments: {len(segments)}\n")
            for seg in segments:
                log.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['speaker']}\n")
        print(f"📝 Log saved: {log_file}")

        return True
    except Exception as e:
        err_msg = f"❌ Diarization test failed: {e}"
        print(err_msg)
        # Log error
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(err_msg + '\n')
        return False

def main():
    """Main test function for Step 2"""
    print("=" * 60)
    print("🧪 Speaker Diarizer - Step 2: HF Auth & Sample Diarization")
    print("=" * 60)
    
    # Step 1: Test imports
    if not test_imports():
        print("❌ Import test failed - stopping here")
        return False
    
    print()
    
    # Step 2: Get HuggingFace token
    hf_token = get_hf_token()
    
    print()
    
    # Step 3: Test pipeline loading with authentication
    success, pipeline = test_pipeline_loading(hf_token)
    if not success:
        print("❌ Pipeline loading failed - stopping here")
        return False
    
    print()
    
    # Step 4: Find sample audio
    audio_file = find_sample_audio()
    
    print()
    
    # Step 5: Test basic diarization if we have everything
    if pipeline and audio_file:
        diarization_success = test_basic_diarization(pipeline, audio_file)
        if not diarization_success:
            print("❌ Diarization test failed")
            return False
    else:
        print("⚠️  Skipping diarization test (missing authentication or audio file)")
    
    print()
    print("🎉 Step 2 completed successfully!")
    print("✅ Ready for Step 3: Advanced diarization features")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
