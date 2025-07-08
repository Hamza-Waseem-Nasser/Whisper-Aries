#!/usr/bin/env python3
"""
Whisper Large-v3 Transcription Script
Optimized for Arabic and English audio files with INT8 quantization
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
from faster_whisper import WhisperModel

class WhisperTranscriber:
    """Whisper transcriber optimized for Arabic and English"""
    
    def __init__(self, 
                 model_size: str = "large-v3",
                 device: str = "auto",
                 compute_type: str = "int8",
                 languages: List[str] = ["ar", "en"]):
        """
        Initialize the transcriber
        
        Args:
            model_size: Whisper model size (large-v3 recommended)
            device: Device to use ("cuda", "cpu", or "auto")
            compute_type: Quantization type ("int8", "float16", "float32")
            languages: Preferred languages for detection
        """
        self.model_size = model_size
        self.languages = languages
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Adjust compute type based on device
        if self.device == "cpu" and compute_type == "float16":
            self.compute_type = "int8"  # CPU doesn't support float16
        else:
            self.compute_type = compute_type
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        print(f"🔄 Loading Whisper {self.model_size} model...")
        print(f"   Device: {self.device}")
        print(f"   Compute type: {self.compute_type}")
        
        try:
            start_time = time.time()
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
            # Display memory usage if using GPU
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"   GPU memory used: {memory_used:.1f} GB")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def transcribe_file(self, 
                       audio_path: str,
                       language: Optional[str] = None,
                       output_format: str = "json") -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            output_format: Output format ("json", "txt", "srt")
            
        Returns:
            Dictionary with transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"🎵 Transcribing: {Path(audio_path).name}")
        
        start_time = time.time()
        
        try:
            # Transcribe with optimized settings
            # Try with VAD filter first, fallback without it if needed
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    temperature=0.0,
                    vad_filter=True,  # Voice Activity Detection
                    word_timestamps=True
                )
            except Exception as vad_error:
                if "VAD" in str(vad_error) or "onnxruntime" in str(vad_error):
                    print("⚠️  VAD filter not available, proceeding without it...")
                    segments, info = self.model.transcribe(
                        audio_path,
                        language=language,
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=False,  # Disable VAD filter
                        word_timestamps=True
                    )
                else:
                    raise vad_error
            
            # Collect results
            results = {
                "file": audio_path,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": []
            }
            
            # Process segments
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                }
                
                # Add word-level timestamps if available
                if segment.words:
                    for word in segment.words:
                        segment_data["words"].append({
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability
                        })
                
                results["segments"].append(segment_data)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            
            # Calculate speed ratio
            speed_ratio = info.duration / processing_time
            results["speed_ratio"] = speed_ratio
            
            print(f"✅ Completed in {processing_time:.2f}s")
            print(f"   Language: {info.language} ({info.language_probability:.2%})")
            print(f"   Duration: {info.duration:.1f}s")
            print(f"   Speed: {speed_ratio:.1f}x real-time")
            
            return results
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            raise
    
    def transcribe_batch(self, 
                        audio_files: List[str],
                        output_dir: str = "transcriptions") -> List[Dict]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save transcriptions
            
        Returns:
            List of transcription results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        print(f"🔄 Transcribing {len(audio_files)} files...")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {Path(audio_file).name}")
            
            try:
                # Transcribe file
                result = self.transcribe_file(audio_file)
                results.append(result)
                
                # Save individual result
                output_file = Path(output_dir) / f"{Path(audio_file).stem}_transcription.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Saved: {output_file}")
                
            except Exception as e:
                print(f"❌ Failed to process {audio_file}: {e}")
                results.append({"file": audio_file, "error": str(e)})
        
        return results
    
    def save_text_output(self, results: Dict, output_path: str):
        """Save transcription as plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in results["segments"]:
                f.write(f"{segment['text']}\n")
    
    def save_srt_output(self, results: Dict, output_path: str):
        """Save transcription as SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(results["segments"], 1):
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file> [output_format]")
        print("Supported formats: json, txt, srt")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "json"
    
    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Transcribe file
        results = transcriber.transcribe_file(audio_file, output_format=output_format)
        
        # Save results
        output_path = Path(audio_file).stem + f"_transcription.{output_format}"
        
        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif output_format == "txt":
            transcriber.save_text_output(results, output_path)
        elif output_format == "srt":
            transcriber.save_srt_output(results, output_path)
        
        print(f"💾 Transcription saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
