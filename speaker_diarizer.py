#!/usr/bin/env python3
"""
Speaker Diarization Module
Provides speaker diarization (who spoke when) for conversation analysis
Optimized for RTX 3050 4GB systems - runs on CPU to preserve GPU VRAM for Whisper
"""

import os
import sys
import time
import warnings
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """A segment with speaker information"""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0

@dataclass
class DiarizationResult:
    """Result from speaker diarization"""
    segments: List[SpeakerSegment]
    num_speakers: int
    processing_time: float
    total_duration: float
    success: bool
    error: str = ""
    metadata: Dict = None

class SpeakerDiarizer:
    """
    CPU-based speaker diarization using pyannote.audio
    Designed to work alongside GPU-based Whisper transcription
    """
    
    def __init__(self, 
                 device: str = "cpu",
                 chunk_duration_minutes: float = 10.0,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None):
        """
        Initialize the speaker diarizer
        
        Args:
            device: Device to use ("cpu" recommended for RTX 3050)
            chunk_duration_minutes: Process audio in chunks of this duration
            min_speakers: Minimum number of speakers (None = auto-detect)
            max_speakers: Maximum number of speakers (None = auto-detect)
        """
        self.device = device
        self.chunk_duration_minutes = chunk_duration_minutes
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        self.pipeline = None
        self._initialized = False
        
        print(f"🎭 Speaker Diarizer Configuration:")
        print(f"   🖥️  Device: {device}")
        print(f"   ⏱️  Chunk duration: {chunk_duration_minutes} minutes")
        print(f"   👥 Speaker range: {min_speakers or 'auto'} - {max_speakers or 'auto'}")
    
    def _initialize_pipeline(self):
        """Initialize the diarization pipeline (lazy loading)"""
        if self._initialized:
            return
        
        try:
            print("🔄 Loading speaker diarization models...")
            start_time = time.time()
            
            from pyannote.audio import Pipeline
            import torch
            
            # Use pre-trained speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None  # This is a public model
            )
            
            # Move to specified device
            if self.device == "cpu":
                self.pipeline = self.pipeline.to(torch.device("cpu"))
            
            load_time = time.time() - start_time
            print(f"✅ Diarization models loaded in {load_time:.1f}s")
            self._initialized = True
            
        except Exception as e:
            print(f"❌ Failed to load diarization models: {e}")
            print(f"💡 Tip: Make sure you have installed: pip install pyannote.audio")
            print(f"💡 You may need to accept the model license on HuggingFace Hub")
            raise
    
    def diarize_file(self, 
                     audio_path: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file
        
        Args:
            audio_path: Path to audio file
            output_dir: Optional directory to save diarization results
        
        Returns:
            DiarizationResult with speaker segments and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                processing_time=0.0,
                total_duration=0.0,
                success=False,
                error=f"Audio file not found: {audio_path}"
            )
        
        print(f"🎭 Starting speaker diarization: {audio_path.name}")
        start_time = time.time()
        
        try:
            # Initialize pipeline if needed
            self._initialize_pipeline()
            
            # Load and process audio
            result = self._process_audio(audio_path)
            
            # Save results if output directory specified
            if output_dir and result.success:
                self._save_diarization_results(result, audio_path, output_dir)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            print(f"✅ Diarization completed in {processing_time:.1f}s")
            print(f"👥 Found {result.num_speakers} speakers in {result.total_duration/60:.1f} minutes")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Diarization failed after {processing_time:.1f}s: {e}")
            
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                processing_time=processing_time,
                total_duration=0.0,
                success=False,
                error=str(e)
            )
    
    def _process_audio(self, audio_path: Path) -> DiarizationResult:
        """Process audio file for speaker diarization"""
        try:
            # Get audio duration first
            duration = self._get_audio_duration(audio_path)
            
            # For very long files, we might need chunked processing
            # For now, let's process the whole file (pyannote handles memory well)
            segments = self._diarize_audio_file(audio_path)
            
            # Count unique speakers
            speakers = set(seg.speaker for seg in segments)
            num_speakers = len(speakers)
            
            return DiarizationResult(
                segments=segments,
                num_speakers=num_speakers,
                processing_time=0.0,  # Will be set by caller
                total_duration=duration,
                success=True,
                metadata={
                    "audio_file": str(audio_path),
                    "processing_method": "full_file",
                    "model": "pyannote/speaker-diarization-3.1"
                }
            )
            
        except Exception as e:
            raise Exception(f"Audio processing failed: {e}")
    
    def _diarize_audio_file(self, audio_path: Path) -> List[SpeakerSegment]:
        """Run diarization on audio file"""
        try:
            # Apply diarization pipeline
            diarization = self.pipeline(str(audio_path))
            
            # Convert to our format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=1.0  # pyannote doesn't provide confidence scores directly
                )
                segments.append(segment)
            
            # Sort by start time
            segments.sort(key=lambda x: x.start)
            
            return segments
            
        except Exception as e:
            raise Exception(f"Diarization pipeline failed: {e}")
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds"""
        try:
            import librosa
            # Just get duration without loading full audio
            duration = librosa.get_duration(filename=str(audio_path))
            return duration
        except ImportError:
            try:
                import soundfile as sf
                with sf.SoundFile(str(audio_path)) as f:
                    return len(f) / f.samplerate
            except ImportError:
                # Fallback: estimate from file size (very rough)
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                # Rough estimate: 1MB ≈ 1 minute for compressed audio
                return file_size_mb * 60
    
    def _save_diarization_results(self, 
                                 result: DiarizationResult, 
                                 audio_path: Path, 
                                 output_dir: Union[str, Path]):
        """Save diarization results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = audio_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save as JSON
            import json
            json_file = output_dir / f"{base_name}_diarization_{timestamp}.json"
            
            diarization_data = {
                "audio_file": str(audio_path),
                "processing_time": result.processing_time,
                "total_duration": result.total_duration,
                "num_speakers": result.num_speakers,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "speaker": seg.speaker,
                        "confidence": seg.confidence
                    }
                    for seg in result.segments
                ],
                "metadata": result.metadata or {}
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(diarization_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Diarization saved: {json_file}")
            
            # Save as simple text format
            txt_file = output_dir / f"{base_name}_speakers_{timestamp}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Speaker Diarization Results\n")
                f.write(f"Audio: {audio_path.name}\n")
                f.write(f"Duration: {result.total_duration/60:.1f} minutes\n")
                f.write(f"Speakers: {result.num_speakers}\n")
                f.write(f"Processing time: {result.processing_time:.1f}s\n")
                f.write(f"\n{'='*50}\n\n")
                
                for i, segment in enumerate(result.segments, 1):
                    start_time = self._format_time(segment.start)
                    end_time = self._format_time(segment.end)
                    duration = segment.end - segment.start
                    f.write(f"Segment {i:3d}: {start_time} - {end_time} "
                           f"({duration:5.1f}s) | {segment.speaker}\n")
            
            print(f"💾 Speaker timeline saved: {txt_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving diarization results: {e}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def align_with_transcription(self, 
                               diarization_result: DiarizationResult,
                               transcription_segments: List[Dict]) -> List[Dict]:
        """
        Align speaker diarization with transcription segments
        
        Args:
            diarization_result: Result from speaker diarization
            transcription_segments: Segments from Whisper transcription
        
        Returns:
            List of transcription segments with speaker information added
        """
        if not diarization_result.success or not transcription_segments:
            return transcription_segments
        
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            # Find overlapping speaker segments
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', trans_start)
            
            # Find the speaker segment with maximum overlap
            best_speaker = "Unknown"
            max_overlap = 0
            
            for spk_seg in diarization_result.segments:
                overlap_start = max(trans_start, spk_seg.start)
                overlap_end = min(trans_end, spk_seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = spk_seg.speaker
            
            # Add speaker information to transcription segment
            aligned_segment = trans_seg.copy()
            aligned_segment['speaker'] = best_speaker
            aligned_segment['speaker_confidence'] = min(1.0, max_overlap / (trans_end - trans_start))
            
            aligned_segments.append(aligned_segment)
        
        return aligned_segments

def main():
    """Main function for testing the diarizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker Diarization Tool")
    parser.add_argument("audio_file", help="Path to audio/video file")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to use (default: cpu)")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return 1
    
    print("🎭 Speaker Diarization Tool")
    print("=" * 50)
    
    # Create diarizer
    diarizer = SpeakerDiarizer(
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    
    # Run diarization
    result = diarizer.diarize_file(
        audio_path,
        output_dir=args.output_dir
    )
    
    if result.success:
        print("🎉 Speaker diarization completed successfully!")
        print(f"👥 Identified {result.num_speakers} speakers")
        return 0
    else:
        print(f"❌ Diarization failed: {result.error}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
