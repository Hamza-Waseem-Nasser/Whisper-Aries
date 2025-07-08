#!/usr/bin/env python3
"""
Automated setup script for Whisper Large-v3 transcription environment
This script installs packages in the correct order to avoid conflicts
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def main():
    """Main setup function"""
    print("🚀 WHISPER LARGE-V3 ENVIRONMENT SETUP")
    print("=" * 60)
    print("This script will install all required packages in the correct order.")
    print("Make sure you have activated your virtual environment first!")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  WARNING: You don't appear to be in a virtual environment!")
        print("   It's recommended to create and activate a virtual environment first:")
        print("   python -m venv whisper-env")
        print("   .\\whisper-env\\Scripts\\activate  # Windows")
        print("   source whisper-env/bin/activate  # Linux/Mac")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return 1
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        print("Failed to upgrade pip, but continuing...")
    
    # Installation steps in correct order
    steps = [
        {
            "command": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "description": "Installing PyTorch with CUDA 12.1 support (CRITICAL - must be first)"
        },
        {
            "command": "pip install faster-whisper",
            "description": "Installing faster-whisper (includes CTranslate2)"
        },
        {
            "command": "pip install onnxruntime-gpu",
            "description": "Installing ONNX Runtime with GPU acceleration"
        },
        {
            "command": "pip install librosa soundfile scipy av",
            "description": "Installing audio processing libraries"
        },
        {
            "command": "pip install huggingface-hub tokenizers tqdm numba matplotlib",
            "description": "Installing supporting utilities"
        }
    ]
    
    failed_steps = []
    
    for i, step in enumerate(steps, 1):
        print(f"\n📦 STEP {i}/{len(steps)}")
        if not run_command(step["command"], step["description"]):
            failed_steps.append(step["description"])
    
    # Verification
    print(f"\n🔍 VERIFICATION")
    print("=" * 40)
    
    verification_commands = [
        ("python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"", "PyTorch installation"),
        ("python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"", "CUDA availability"),
        ("python -c \"import faster_whisper; print('faster-whisper: OK')\"", "faster-whisper installation"),
        ("python -c \"import onnxruntime; print('onnxruntime-gpu: OK')\"", "ONNX Runtime installation"),
        ("python -c \"import librosa; print('librosa: OK')\"", "Audio libraries"),
    ]
    
    verification_failed = []
    for cmd, desc in verification_commands:
        if not run_command(cmd, f"Verifying {desc}"):
            verification_failed.append(desc)
    
    # Final results
    print(f"\n" + "=" * 60)
    if not failed_steps and not verification_failed:
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("✅ All packages installed successfully!")
        print("✅ Environment is ready for Whisper Large-v3 transcription!")
        print()
        print("📋 NEXT STEPS:")
        print("1. Run verification: python verify_setup.py")
        print("2. Test transcription: python demo.py")
        print("3. Use the transcriber: python transcribe.py <audio_file>")
        return 0
    else:
        print("❌ SETUP ISSUES DETECTED")
        print("=" * 60)
        if failed_steps:
            print("Failed installation steps:")
            for step in failed_steps:
                print(f"   - {step}")
        if verification_failed:
            print("Failed verification steps:")
            for step in verification_failed:
                print(f"   - {step}")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("1. Make sure you have Python 3.8+ installed")
        print("2. Check that you have CUDA 12.1 drivers installed")
        print("3. Try running individual commands manually")
        print("4. Check docs/troubleshooting.md for more help")
        return 1

if __name__ == "__main__":
    sys.exit(main())
