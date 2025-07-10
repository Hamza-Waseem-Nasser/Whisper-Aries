"""
conversation_analyzer.py - Aligns transcription and diarization results

- Loads transcription and diarization segments
- Aligns them using overlap/confidence logic from utils
- Handles multilingual and code-switching content
- Outputs structured conversation data
"""

from typing import List, Dict, Any
from utils import align_segments, ConversationError
import config

class ConversationAnalyzer:
    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = confidence_threshold or config.DEFAULT_CONFIDENCE_THRESHOLD

    def analyze(self, transcription_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
        """
        Aligns transcription and diarization segments, returns speaker-attributed transcript.
        """
        try:
            aligned = align_segments(transcription_segments, diarization_segments, confidence_threshold=self.confidence_threshold)
            # Optionally: add language detection, validation, etc.
            return aligned
        except Exception as e:
            raise ConversationError(f"Alignment failed: {e}")

# --- Example usage (to be replaced with actual pipeline integration) ---
if __name__ == "__main__":
    # Placeholder: load segments from files or pipeline
    transcription_segments = [
        {'text': 'Hello.', 'start': 0.0, 'end': 1.5},
        {'text': 'How are you?', 'start': 1.6, 'end': 3.0},
    ]
    diarization_segments = [
        {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 2.0},
        {'speaker': 'SPEAKER_01', 'start': 2.0, 'end': 4.0},
    ]
    analyzer = ConversationAnalyzer()
    aligned = analyzer.analyze(transcription_segments, diarization_segments)
    for seg in aligned:
        print(seg)
