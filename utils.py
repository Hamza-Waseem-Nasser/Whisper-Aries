"""
utils.py - Core utilities for conversation analysis pipeline

Includes:
- Audio preprocessing helpers
- Timestamp and segment overlap calculations
- Memory management helpers
- Custom error classes
"""


import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

# --- Error Classes ---
class ConversationError(Exception):
    """Base exception for conversation processing"""
    pass

class DiarizationError(ConversationError):
    """Diarization-specific errors"""
    pass

class TranscriptionError(ConversationError):
    """Transcription-specific errors"""
    pass

# --- Timestamp/Segment Utilities ---
def segment_overlap(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    """
    Calculate overlap duration between two segments (start, end).
    Returns overlap in seconds (0 if no overlap).
    """
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    return max(0.0, end - start)

def align_segments(transcription_segments: List[Dict], diarization_segments: List[Dict], confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Align transcription segments with diarization segments using overlap and confidence scoring.
    Returns a list of dicts with text, speaker, start, end, and confidence.
    """
    aligned = []
    for t in transcription_segments:
        t_start, t_end = t['start'], t['end']
        overlaps = []
        for d in diarization_segments:
            d_start, d_end = d['start'], d['end']
            overlap = segment_overlap((t_start, t_end), (d_start, d_end))
            if overlap > 0:
                overlaps.append((overlap, d['speaker']))
        if overlaps:
            total = sum([o[0] for o in overlaps])
            speaker_scores = {}
            for overlap, speaker in overlaps:
                speaker_scores[speaker] = speaker_scores.get(speaker, 0) + overlap
            # Majority-overlap rule
            best_speaker = max(speaker_scores, key=speaker_scores.get)
            confidence = speaker_scores[best_speaker] / total if total > 0 else 0.0
            if confidence >= confidence_threshold:
                assigned_speaker = best_speaker
            else:
                assigned_speaker = None
        else:
            assigned_speaker = None
            confidence = 0.0
        aligned.append({
            'text': t['text'],
            'start': t_start,
            'end': t_end,
            'speaker': assigned_speaker,
            'confidence': confidence
        })
    return aligned

# --- Memory Management Helpers ---
def get_memory_usage() -> Optional[float]:
    """
    Returns current process memory usage in MB (if psutil is available).
    """
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024)
        return mem
    except ImportError:
        return None

# --- Audio Extraction Utility ---
import tempfile
import subprocess
import shutil

def extract_audio_if_needed(input_path: str, preferred_ext: str = ".wav", temp_dir: str = None) -> str:
    """
    If input_path is a video or unsupported audio, extract audio to .wav using ffmpeg.
    Returns path to audio file (original or extracted). Handles temp file cleanup.
    """
    supported_audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(input_path).suffix.lower()
    if ext in supported_audio_exts:
        return input_path  # Already audio

    # Extract audio
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio extraction but was not found in PATH.")

    temp_dir = temp_dir or tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=preferred_ext, dir=temp_dir, delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # uncompressed wav
        "-ar", "16000",  # 16kHz for ASR/diarization
        "-ac", "1",      # mono
        tmp_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1024:
            raise RuntimeError(f"Audio extraction failed or output too small: {tmp_path}")
        return tmp_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"ffmpeg audio extraction failed: {e}\n{getattr(e, 'stderr', b'').decode(errors='ignore')}")
