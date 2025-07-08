#!/usr/bin/env python3
"""
Whisper Large-v3 Setup Verification Script
Verifies installation and downloads the model for first-time setup
"""

import sys
import time
import torch
from pathlib import Path
import config

def verify_pytorch_cuda():
    """Verify PyTorch and CUDA setup"""
    print("🔧 PYTORCH & CUDA VERIFICATION")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return "cuda"
    else:
        print("⚠️  No CUDA GPU detected - will use CPU mode")
        return "cpu"

def verify_faster_whisper():
    """Verify faster-whisper installation and download model"""
    print("\n🎤 FASTER-WHISPER VERIFICATION")
    print("=" * 50)
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper imported successfully")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = config.COMPUTE_TYPE
        
        print(f"\n📥 Downloading/Loading Whisper {config.MODEL_SIZE}...")
        print("   Note: First run will download the model (~3GB)")
        print(f"   Device: {device}")
        print(f"   Compute type: {compute_type}")
        
        # Load model
        start_time = time.time()
        model = WhisperModel(
            config.MODEL_SIZE, 
            device=device, 
            compute_type=compute_type
        )
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU memory used: {memory_used:.1f} GB")
        
        return True, model, device
        
    except ImportError as e:
        print(f"❌ Failed to import faster-whisper: {e}")
        return False, None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False, None, None

def verify_audio_libraries():
    """Verify audio processing libraries"""
    print("\n🎵 AUDIO LIBRARIES VERIFICATION")
    print("=" * 50)
    
    libraries = [
        ("librosa", "Audio analysis and feature extraction"),
        ("soundfile", "Audio file I/O"),
        ("numpy", "Numerical computing"),
        ("scipy", "Scientific computing"),
        ("av", "FFmpeg bindings for video/audio"),
    ]
    
    all_good = True
    for lib_name, description in libraries:
        try:
            __import__(lib_name)
            print(f"✅ {lib_name:12} - {description}")
        except ImportError:
            print(f"❌ {lib_name:12} - {description} (NOT INSTALLED)")
            all_good = False
    
    return all_good

def test_transcription(model, device):
    """Test transcription with a sample audio file if available"""
    print("\n🎵 TRANSCRIPTION TEST")
    print("=" * 50)
    
    # Look for audio files
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(".").glob(f"*{ext}"))
    
    if not audio_files:
        print("📁 No audio files found for testing")
        print("   Place an audio file in this directory to test transcription")
        return True
    
    test_file = audio_files[0]
    print(f"🎵 Testing with: {test_file.name}")
    
    try:
        start_time = time.time()
        segments, info = model.transcribe(
            str(test_file),
            beam_size=config.BEAM_SIZE,
            temperature=config.TEMPERATURE,
            vad_filter=config.VAD_FILTER,
            word_timestamps=config.INCLUDE_WORD_TIMESTAMPS
        )
        
        processing_time = time.time() - start_time
        
        print("✅ Transcription successful!")
        print(f"   🌍 Language: {info.language} ({info.language_probability:.1%})")
        print(f"   ⏱️  Duration: {info.duration:.1f} seconds")
        print(f"   🚀 Processing: {processing_time:.1f} seconds")
        print(f"   ⚡ Speed: {info.duration/processing_time:.1f}x real-time")
        
        # Show first segment
        for i, segment in enumerate(segments):
            if i >= 1:  # Just show first segment
                break
            preview_text = segment.text[:100] + "..." if len(segment.text) > 100 else segment.text
            print(f"   📝 Preview: [{segment.start:.1f}s] {preview_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False

def create_environment_info():
    """Create environment information file"""
    env_info = f"""# Environment Information
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## System
- Python: {sys.version}
- PyTorch: {torch.__version__}
- CUDA Available: {torch.cuda.is_available()}

## Configuration
- Model: {config.MODEL_SIZE}
- Device: {config.DEVICE}
- Compute Type: {config.COMPUTE_TYPE}
- Languages: {config.LANGUAGES}
- VAD Filter: {config.VAD_FILTER}

## GPU Information
"""
    
    if torch.cuda.is_available():
        env_info += f"""- GPU: {torch.cuda.get_device_name(0)}
- CUDA Version: {torch.version.cuda}
- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB
"""
    else:
        env_info += "- No CUDA GPU detected\n"
    
    with open("ENVIRONMENT.md", "w", encoding="utf-8") as f:
        f.write(env_info)
    
    print("📝 Environment information saved to ENVIRONMENT.md")

def main():
    """Main verification function"""
    print("🚀 WHISPER LARGE-V3 SETUP VERIFICATION")
    print("=" * 60)
    
    # Verify PyTorch
    device = verify_pytorch_cuda()
    
    # Verify audio libraries
    audio_libs_ok = verify_audio_libraries()
    
    # Verify faster-whisper and download model
    whisper_ok, model, actual_device = verify_faster_whisper()
    
    # Test transcription if everything is working
    transcription_ok = True
    if whisper_ok and model:
        transcription_ok = test_transcription(model, actual_device)
    
    # Create environment info
    create_environment_info()
    
    # Final result
    print("\n" + "=" * 60)
    if whisper_ok and audio_libs_ok and transcription_ok:
        print("🎉 SUCCESS!")
        print("=" * 60)
        print("✅ All components verified and working correctly!")
        print("✅ Whisper Large-v3 is ready for Arabic and English transcription")
        print("✅ Model downloaded and cached for future use")
        print("\n📋 NEXT STEPS:")
        print("1. Place your audio files in this directory")
        print("2. Run: python transcribe.py <audio_file>")
        print("3. Or run: python demo.py for automatic detection")
        print("4. For batch processing: python batch_transcribe.py")
        
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print("Some components failed verification.")
        print("Check the error messages above and:")
        print("1. Ensure all packages are installed: pip install -r requirements.txt")
        print("2. Check CUDA drivers if using GPU")
        print("3. Try CPU mode if GPU issues persist")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
