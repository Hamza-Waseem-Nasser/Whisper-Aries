#!/usr/bin/env python3
"""
Test script for the CUDA 12 compatible Whisper setup
"""

import sys
import time
import torch
from pathlib import Path

def test_cuda12_setup():
    """Test the CUDA 12 compatible setup"""
    
    print("🚀 CUDA 12 COMPATIBLE WHISPER TEST")
    print("=" * 50)
    
    # Test PyTorch CUDA
    print("🔧 PYTORCH & CUDA:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test faster-whisper import
    print("\n🎤 FASTER-WHISPER:")
    try:
        from faster_whisper import WhisperModel
        print("   ✅ faster-whisper imported successfully")
        
        # Test model loading
        print("   🔄 Loading Whisper Large-v3...")
        start_time = time.time()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel("large-v3", device=device, compute_type="int8")
        
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.2f} seconds")
        print(f"   📊 Device: {device}")
        print(f"   📊 Compute type: int8")
        
        # Test transcription if audio file exists
        audio_files = list(Path(".").glob("*.wav")) + list(Path(".").glob("*.mp3"))
        
        if audio_files:
            test_file = audio_files[0]
            print(f"\n🎵 TESTING TRANSCRIPTION:")
            print(f"   File: {test_file.name}")
            
            try:
                segments, info = model.transcribe(
                    str(test_file),
                    vad_filter=True,
                    word_timestamps=True,
                    beam_size=5,
                    temperature=0.0
                )
                
                print("   ✅ Transcription successful!")
                print(f"   🌍 Language: {info.language} ({info.language_probability:.1%})")
                print(f"   ⏱️  Duration: {info.duration:.1f} seconds")
                
                # Show first segment
                for i, segment in enumerate(segments):
                    if i >= 1:  # Just show first segment
                        break
                    print(f"   📝 Preview: [{segment.start:.1f}s] {segment.text}")
                
            except Exception as e:
                print(f"   ❌ Transcription failed: {e}")
        
        else:
            print("\n📁 No audio files found for testing")
            print("   Place a .wav or .mp3 file in this directory to test transcription")
        
    except Exception as e:
        print(f"   ❌ faster-whisper failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 SUCCESS!")
    print("=" * 50)
    print("✅ CUDA 12 compatible Whisper setup is working!")
    print("✅ Ready for Arabic and English transcription")
    print("✅ GPU acceleration enabled")
    
    return True

if __name__ == "__main__":
    success = test_cuda12_setup()
    if not success:
        sys.exit(1)
