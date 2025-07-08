# Setup Guide

## Prerequisites

- **Python 3.10 or 3.11** (Python 3.12+ not yet supported)
- **Git** for cloning the repository
- **NVIDIA GPU** with CUDA support (optional but recommended)

## Installation Steps

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd whisper-project
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv whisper-env
.\whisper-env\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch with CUDA 12.1 support
- faster-whisper with quantization
- Audio processing libraries
- Supporting utilities

### 4. Verify Installation

```bash
python verify_setup.py
```

This script will:
- Check PyTorch and CUDA installation
- Download Whisper Large-v3 model (~3GB)
- Verify all components are working
- Test transcription if audio files are present

## First Run

The first time you run transcription, the model will be downloaded:

```bash
python transcribe.py sample_audio.wav
```

## Configuration

Edit `config.py` to customize:

```python
# Switch to CPU if no GPU
DEVICE = "cpu"

# Use smaller model for faster loading
MODEL_SIZE = "medium"

# Disable VAD if causing issues
VAD_FILTER = False
```

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues and solutions.
