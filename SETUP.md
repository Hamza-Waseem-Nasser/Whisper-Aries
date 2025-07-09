# Whisper Project: Quick Setup Guide

This guide helps you set up and run the optimized Whisper transcriber on your machine.

## 1. Python Version
- Requires **Python 3.10 or 3.11** (recommended: 3.11)

## 2. Create and Activate Virtual Environment
```powershell
python -m venv whisper-env
# Activate (Windows)
whisper-env\Scripts\activate
# Activate (Linux/Mac)
source whisper-env/bin/activate
```

## 3. Install Requirements
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Download Models (First Run Only)
- The first time you run the transcriber, the Whisper model will be downloaded automatically to the `models/` directory.

## 5. Run the Optimized Transcriber
```powershell
python final_optimized_transcriber.py "your_audio_file.mp3" --model base --workers 2 --language en
```
- Replace `your_audio_file.mp3` with your file.
- For best speed/quality on RTX 3050: use `--model base --workers 2`.
- For highest accuracy: use `--model large-v3 --workers 1`.

## 6. Output
- Transcriptions and reports will be saved in the project directory or `transcriptions/` folder.

---

**No other setup scripts are needed.**

If you need to update requirements, run:
```powershell
pip freeze > requirements.txt
```

For troubleshooting, ensure FFmpeg is installed and available in your PATH (for video/audio support).
