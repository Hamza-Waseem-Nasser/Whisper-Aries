#!/usr/bin/env python3
"""
Automated environment setup script for team deployment
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 10 or version.minor > 11:
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Whisper requires Python 3.10 or 3.11")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
    return True

def setup_environment():
    """Set up the virtual environment and install dependencies"""
    print("🚀 WHISPER PROJECT SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    venv_name = "whisper-env"
    if not run_command(f"python -m venv {venv_name}", "Creating virtual environment"):
        return False
    
    # Determine activation script
    if platform.system() == "Windows":
        activate_script = f"{venv_name}\\Scripts\\activate"
        python_exe = f"{venv_name}\\Scripts\\python.exe"
        pip_exe = f"{venv_name}\\Scripts\\pip.exe"
    else:
        activate_script = f"{venv_name}/bin/activate"
        python_exe = f"{venv_name}/bin/python"
        pip_exe = f"{venv_name}/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{python_exe} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_exe} install -r requirements.txt", "Installing requirements"):
        return False
    
    # Verify installation
    print("\n🔍 VERIFYING INSTALLATION")
    print("=" * 50)
    
    verify_command = f"{python_exe} verify_setup.py"
    if run_command(verify_command, "Running verification"):
        print("\n🎉 SETUP COMPLETE!")
        print("=" * 50)
        print("✅ Environment ready for transcription")
        print(f"\n📋 TO GET STARTED:")
        if platform.system() == "Windows":
            print(f"   1. {venv_name}\\Scripts\\activate")
        else:
            print(f"   1. source {venv_name}/bin/activate")
        print("   2. python transcribe.py <audio_file>")
        print("   3. Or: python demo.py")
        return True
    else:
        print("\n❌ SETUP VERIFICATION FAILED")
        print("Check the error messages above and run verify_setup.py manually")
        return False

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
