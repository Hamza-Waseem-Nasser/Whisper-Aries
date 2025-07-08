# Whisper Configuration
# Production-ready settings for Arabic and English transcription

# Model settings
MODEL_SIZE = "large-v3"      # Options: large-v3, medium, small, base, tiny
DEVICE = "auto"              # Options: auto, cuda, cpu
COMPUTE_TYPE = "int8"        # Options: int8, float16, float32

# Language settings
LANGUAGES = ["ar", "en"]     # Arabic and English
AUTO_DETECT_LANGUAGE = True  # Automatic language detection
LANGUAGE_DETECTION_THRESHOLD = 0.5

# Audio settings
AUDIO_SAMPLE_RATE = 16000    # Sample rate for processing
AUDIO_CHANNELS = 1           # Mono audio

# Output settings
OUTPUT_FORMAT = "json"       # Default: json, txt, srt, vtt
INCLUDE_TIMESTAMPS = True    # Include segment timestamps
INCLUDE_WORD_TIMESTAMPS = True  # Include word-level timestamps

# Performance settings
BEAM_SIZE = 5               # Search beam size (higher = slower but more accurate)
TEMPERATURE = 0.0           # Sampling temperature (0.0 = deterministic)
VAD_FILTER = False          # Voice Activity Detection (disable if causing issues)
VAD_THRESHOLD = 0.5         # VAD sensitivity

# Advanced settings
MAX_SEGMENT_LENGTH = 30     # Maximum segment length in seconds
MIN_SEGMENT_LENGTH = 0.5    # Minimum segment length in seconds
COMPRESSION_RATIO_THRESHOLD = 2.4  # Text compression ratio threshold
LOGPROB_THRESHOLD = -1.0    # Log probability threshold
NO_SPEECH_THRESHOLD = 0.6   # No speech threshold

# File processing
MAX_AUDIO_DURATION = 3600   # Maximum audio duration in seconds (1 hour)
SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".webm"]

# GPU settings
CUDA_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use
