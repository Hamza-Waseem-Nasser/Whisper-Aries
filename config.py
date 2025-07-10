"""
config.py - Centralized configuration for conversation analysis pipeline

Holds model paths, hardware settings, language/output preferences, and defaults.
"""

import os
from pathlib import Path

# --- Model Paths ---
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', str(Path('models/models--Systran--faster-whisper-large-v3')))
DIARIZATION_MODEL_NAME = os.getenv('DIARIZATION_MODEL_NAME', 'pyannote/speaker-diarization-3.1')

# --- Hardware Settings ---
USE_GPU = os.getenv('USE_GPU', 'auto')  # 'auto', 'cpu', 'cuda', or device id
DIARIZATION_DEVICE = os.getenv('DIARIZATION_DEVICE', 'cpu')  # Diarization runs on CPU by default

# --- Language Preferences ---
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'auto')  # 'auto', 'en', 'ar', etc.
SUPPORTED_LANGUAGES = ['en', 'ar']

# --- Output Format Defaults ---
DEFAULT_OUTPUT_FORMATS = os.getenv('DEFAULT_OUTPUT_FORMATS', 'html,json,srt').split(',')
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))
DEFAULT_CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 30))  # in seconds

# --- Other Settings ---
PROGRESS_BAR = os.getenv('PROGRESS_BAR', '1') == '1'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# --- Metadata ---
PIPELINE_VERSION = '1.0.0'

# --- Output Directories ---
OUTPUT_DIR = os.getenv('OUTPUT_DIR', str(Path('transcriptions/transcriptions')))

# --- HuggingFace Token ---
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

# --- Utility ---
def print_config():
    print(f"Whisper model path: {WHISPER_MODEL_PATH}")
    print(f"Diarization model: {DIARIZATION_MODEL_NAME}")
    print(f"Use GPU: {USE_GPU}")
    print(f"Diarization device: {DIARIZATION_DEVICE}")
    print(f"Default language: {DEFAULT_LANGUAGE}")
    print(f"Output formats: {DEFAULT_OUTPUT_FORMATS}")
    print(f"Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")
    print(f"Chunk size: {DEFAULT_CHUNK_SIZE}s")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Log level: {LOG_LEVEL}")
    print(f"Pipeline version: {PIPELINE_VERSION}")
