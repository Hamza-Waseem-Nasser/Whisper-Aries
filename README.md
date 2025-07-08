# Whisper Large-v3 Transcription System

A production-ready Arabic/English transcription system using OpenAI's Whisper Large-v3 with GPU acceleration and INT8 quantization.

## 🎯 Features

- **Whisper Large-v3** with INT8 quantization for speed
- **GPU acceleration** (CUDA) with CPU fallback
- **Arabic & English** optimized transcription
- **Multiple output formats**: JSON, TXT, SRT subtitles
- **Batch processing** for multiple files
- **Word-level timestamps** for precise alignment
- **Voice Activity Detection** (VAD) support

## 🔧 Quick Setup

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd whisper-project
python -m venv whisper-env
```

**Windows:**
```powershell
.\whisper-env\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source whisper-env/bin/activate
```

### 2. Install Dependencies

**Option A: Automated Installation (Recommended)**
```bash
python install_packages.py
```

**Option B: Manual Installation**
```bash
pip install --upgrade pip
# IMPORTANT: Install PyTorch with CUDA 12.1 support FIRST
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Then install other packages
pip install faster-whisper
pip install onnxruntime-gpu
pip install librosa soundfile scipy av
pip install huggingface-hub tokenizers tqdm numba matplotlib
```

**Option C: Requirements File (may have dependency conflicts)**
```bash
pip install -r requirements.txt
```

⚠️ **IMPORTANT**: Package installation order matters! Use Option A for best results.

### 3. Verify Installation

```bash
python verify_setup.py
```

## 🚀 Quick Start

### Single File Transcription

```bash
# Basic usage
python transcribe.py audio_file.mp3

# With specific output format
python transcribe.py audio_file.mp3 --format json    # Detailed JSON
python transcribe.py audio_file.mp3 --format txt     # Plain text
python transcribe.py audio_file.mp3 --format srt     # Subtitles
```

### Batch Processing

```bash
# Process all audio files in current directory
python batch_transcribe.py

# Process specific directory
python batch_transcribe.py --input-dir "path/to/audio" --output-dir "path/to/results"
```

### Demo Script

```bash
# Auto-detect and transcribe audio files
python demo.py
```

## 📁 Project Structure

```
whisper-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── transcribe.py               # Main transcription script
├── batch_transcribe.py         # Batch processing
├── demo.py                     # Simple demo
├── verify_setup.py             # Installation verification
├── examples/                   # Example audio files
│   └── sample_audio.wav
├── docs/                       # Documentation
│   ├── setup_guide.md
│   ├── troubleshooting.md
│   └── api_reference.md
└── scripts/                    # Utility scripts
    ├── setup_environment.py
    └── benchmark.py
```

## ⚡ Performance

- **GPU Mode**: ~5-10x faster than real-time
- **CPU Mode**: ~2-3x faster than real-time  
- **Memory Usage**: ~3-4GB VRAM (GPU) or ~4-6GB RAM (CPU)
- **Model Size**: ~3GB download (first run only)

## 🌍 Language Support

- **Arabic**: Excellent accuracy with proper UTF-8 encoding
- **English**: Native support with high accuracy
- **Auto-detection**: Automatic language identification
- **Mixed content**: Handles Arabic-English mixed audio

## 📊 Output Formats

### JSON (Detailed)
```json
{
  "language": "ar",
  "confidence": 0.99,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "مرحبا بكم",
      "words": [...]
    }
  ]
}
```

### TXT (Plain Text)
```
مرحبا بكم في هذا البرنامج
Welcome to this program
```

### SRT (Subtitles)
```
1
00:00:00,000 --> 00:00:03,500
مرحبا بكم في هذا البرنامج

2
00:00:03,500 --> 00:00:07,000
Welcome to this program
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_SIZE = "large-v3"      # large-v3, medium, small
DEVICE = "auto"              # auto, cuda, cpu
COMPUTE_TYPE = "int8"        # int8, float16, float32

# Languages
LANGUAGES = ["ar", "en"]     # Arabic, English
AUTO_DETECT = True

# Performance
VAD_FILTER = True           # Voice Activity Detection
BEAM_SIZE = 5               # Search beam size
TEMPERATURE = 0.0           # Sampling temperature
```

## 🛠️ Troubleshooting

### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Switch to CPU mode: Set `DEVICE = "cpu"` in config.py
- Use smaller model: Set `MODEL_SIZE = "medium"`
- Reduce batch size for multiple files

### Audio Format Issues
- Supported: MP3, WAV, M4A, FLAC, OGG, MP4, WEBM
- Convert unsupported formats to WAV/MP3 first

## 📈 System Requirements

### Minimum
- **OS**: Windows 10+, Linux, macOS
- **Python**: 3.10 or 3.11
- **RAM**: 8GB
- **Storage**: 10GB free space

### Recommended (GPU)
- **GPU**: NVIDIA RTX 3050+ with 4GB+ VRAM
- **CUDA**: 11.8+ or 12.x
- **RAM**: 16GB
- **Storage**: SSD with 20GB+ free space

## 🔄 Updates

```bash
# Update packages
pip install --upgrade faster-whisper torch torchaudio

# Update repository
git pull origin main
```

## 📞 Support

- Check `docs/troubleshooting.md` for common issues
- Run `python verify_setup.py` to diagnose problems
- Create GitHub issues for bugs or feature requests

## 📄 License

MIT License - see LICENSE file for details

---

**Ready to transcribe!** 🎤 ➡️ 📝
