# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. PyTorch CUDA Installation Failed

**Problem:** CUDA version mismatch or GPU not detected

**Solutions:**
```bash
# Check GPU and CUDA
nvidia-smi

# Install CPU version if GPU issues persist
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. faster-whisper Import Error

**Problem:** `ImportError: No module named 'faster_whisper'`

**Solution:**
```bash
pip install --upgrade faster-whisper
```

#### 3. CUDA Library Errors (cublas64_12.dll)

**Problem:** Missing CUDA libraries

**Solutions:**
```bash
# Option 1: Install NVIDIA CUDA libraries
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

# Option 2: Use CPU mode
# Edit config.py: DEVICE = "cpu"

# Option 3: Downgrade CTranslate2
pip install ctranslate2==3.24.0
```

### Runtime Issues

#### 1. Out of Memory (GPU)

**Problem:** GPU runs out of memory

**Solutions:**
```python
# Edit config.py
DEVICE = "cpu"  # Switch to CPU
MODEL_SIZE = "medium"  # Use smaller model
COMPUTE_TYPE = "int8"  # Use quantization
```

#### 2. Slow Transcription

**Problem:** Transcription is very slow

**Solutions:**
```python
# Edit config.py
DEVICE = "cuda"  # Use GPU if available
COMPUTE_TYPE = "int8"  # Use quantization
VAD_FILTER = True  # Enable voice activity detection
BEAM_SIZE = 1  # Reduce beam size for speed
```

#### 3. Poor Transcription Quality

**Problem:** Inaccurate transcription

**Solutions:**
```python
# Edit config.py
COMPUTE_TYPE = "float16"  # Higher precision
BEAM_SIZE = 5  # Increase beam size
TEMPERATURE = 0.0  # Deterministic output
```

#### 4. Audio Format Not Supported

**Problem:** Audio file cannot be processed

**Solutions:**
```bash
# Convert to supported format using FFmpeg
ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav

# Supported formats: MP3, WAV, M4A, FLAC, OGG, MP4, WEBM
```

### Language-Specific Issues

#### 1. Arabic Text Display Issues

**Problem:** Arabic text appears as boxes or incorrect characters

**Solutions:**
- Ensure terminal/editor supports UTF-8
- Use JSON output format for proper encoding
- On Windows: `chcp 65001` for UTF-8 console

#### 2. Language Detection Issues

**Problem:** Wrong language detected

**Solutions:**
```python
# Force specific language
segments, info = model.transcribe(
    audio_file,
    language="ar"  # or "en"
)
```

### Performance Optimization

#### 1. Batch Processing Optimization

```python
# Process multiple files efficiently
for audio_file in audio_files:
    segments, info = model.transcribe(
        audio_file,
        vad_filter=True,  # Skip silence
        word_timestamps=False  # Faster processing
    )
```

#### 2. Memory Management

```python
# Clear GPU memory between files
import torch
torch.cuda.empty_cache()
```

### Environment Issues

#### 1. Virtual Environment Problems

**Problem:** Packages not found despite installation

**Solutions:**
```bash
# Recreate virtual environment
deactivate
rm -rf whisper-env  # or Remove-Item on Windows
python -m venv whisper-env
source whisper-env/bin/activate  # or .\whisper-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Python Version Compatibility

**Problem:** Unsupported Python version

**Solutions:**
- Use Python 3.10 or 3.11
- Python 3.12+ not yet fully supported

### Getting Help

1. Run `python verify_setup.py` to diagnose issues
2. Check the generated `ENVIRONMENT.md` file
3. Create GitHub issue with:
   - Error message
   - Environment information
   - Steps to reproduce

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Reset to Defaults

If all else fails:

```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/faster-whisper/

# Reinstall from scratch
pip uninstall -y faster-whisper ctranslate2 torch torchaudio
pip install -r requirements.txt
```
