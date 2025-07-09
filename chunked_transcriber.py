#!/usr/bin/env python3
"""
Chunked/Parallel Whisper transcription for fast processing of large files
"""

import os
import sys
import time
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue
import threading

try:
    from pydub import AudioSegment
    import numpy as np
    # Try faster-whisper first, fallback to regular whisper
    try:
        from faster_whisper import WhisperModel
        FASTER_WHISPER_AVAILABLE = True
    except ImportError:
        FASTER_WHISPER_AVAILABLE = False
    
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
    
    if not FASTER_WHISPER_AVAILABLE and not WHISPER_AVAILABLE:
        raise ImportError("Neither faster-whisper nor openai-whisper is available")
        
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please install required packages:")
    print("pip install pydub")
    if not FASTER_WHISPER_AVAILABLE:
        print("pip install faster-whisper")
    sys.exit(1)

class ChunkedWhisperTranscriber:
    """
    Fast chunked/parallel transcription for large audio files
    """
    
    def __init__(self, model_size="large-v3", device="auto", compute_type="int8", 
                 chunk_length_minutes=5, overlap_seconds=2, max_workers=None):
        """
        Initialize the chunked transcriber
        
        Args:
            model_size: Whisper model to use
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute type for faster-whisper
            chunk_length_minutes: Length of each chunk in minutes
            overlap_seconds: Overlap between chunks to avoid word cuts
            max_workers: Max parallel workers (None = auto)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.chunk_length_minutes = chunk_length_minutes
        self.overlap_seconds = overlap_seconds
        self.max_workers = max_workers or min(4, os.cpu_count())
        
        self.model = None
        self.progress_lock = Lock()
        self.progress_callback = None
        
        print(f"🔧 Chunked Transcriber Config:")
        print(f"   📏 Chunk length: {chunk_length_minutes} minutes")
        print(f"   🔄 Overlap: {overlap_seconds} seconds")
        print(f"   👥 Max workers: {self.max_workers}")
    
    def load_model(self):
        """Load the Whisper model"""
        if self.model is None:
            print(f"🤖 Loading Whisper {self.model_size} model...")
            try:
                # Try faster-whisper first
                if FASTER_WHISPER_AVAILABLE:
                    self.model = WhisperModel(
                        self.model_size, 
                        device=self.device,
                        compute_type=self.compute_type
                    )
                    self.model_type = "faster-whisper"
                    print(f"✅ Loaded faster-whisper model on {self.device}")
                elif WHISPER_AVAILABLE:
                    # Fallback to standard whisper
                    self.model = whisper.load_model(self.model_size, device=self.device)
                    self.model_type = "whisper"
                    print(f"✅ Loaded standard whisper model on {self.device}")
                else:
                    raise ImportError("No whisper implementation available")
                    
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
    
    def get_audio_duration(self, audio_path):
        """Get audio duration using multiple methods"""
        try:
            # Method 1: Try pydub
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert to seconds
        except Exception as e1:
            print(f"⚠️  Pydub failed, trying alternative method: {e1}")
            
            try:
                # Method 2: Try with librosa if available
                import librosa
                y, sr = librosa.load(audio_path)
                return len(y) / sr
            except ImportError:
                pass
            except Exception as e2:
                print(f"⚠️  Librosa failed: {e2}")
            
            try:
                # Method 3: Estimate from file size (very rough)
                file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
                # Rough estimate: MP3 at 128kbps ≈ 1MB per minute
                estimated_duration = file_size_mb * 60  # seconds
                print(f"⚠️  Using rough duration estimate: {estimated_duration/60:.1f} minutes")
                return estimated_duration
            except Exception as e3:
                print(f"⚠️  File size estimation failed: {e3}")
                
            # Method 4: Use faster-whisper to get duration from a quick transcription
            try:
                print("⚠️  Attempting to get duration from Whisper model...")
                # Load model if not already loaded
                if self.model is None:
                    self.load_model()
                
                if self.model_type == "faster-whisper":
                    # Quick transcription to get info
                    segments, info = self.model.transcribe(str(audio_path), beam_size=1)
                    # Convert generator to list to get last segment
                    segment_list = list(segments)
                    if segment_list:
                        return segment_list[-1].end
                
                # Fallback: assume 60 seconds for testing
                print("⚠️  Using fallback duration of 60 seconds")
                return 60.0
                
            except Exception as e4:
                print(f"⚠️  Whisper duration detection failed: {e4}")
                return None
    
    def split_audio_into_chunks(self, audio_path, output_dir):
        """
        Split audio file into overlapping chunks
        
        Returns:
            List of (chunk_file_path, start_time, end_time) tuples
        """
        print(f"🔪 Splitting audio into {self.chunk_length_minutes}-minute chunks...")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000.0
        
        chunk_length_ms = self.chunk_length_minutes * 60 * 1000
        overlap_ms = self.overlap_seconds * 1000
        
        chunks = []
        start_ms = 0
        chunk_num = 0
        
        while start_ms < len(audio):
            # Calculate end time for this chunk
            end_ms = min(start_ms + chunk_length_ms, len(audio))
            
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            
            # Save chunk
            chunk_file = output_dir / f"chunk_{chunk_num:03d}.wav"
            chunk.export(str(chunk_file), format="wav")
            
            chunks.append((
                chunk_file,
                start_ms / 1000.0,  # start time in seconds
                end_ms / 1000.0     # end time in seconds
            ))
            
            chunk_num += 1
            
            # Move start position (with overlap)
            if end_ms == len(audio):
                break
            start_ms = end_ms - overlap_ms
            
            # Ensure we don't go negative
            if start_ms < 0:
                start_ms = 0
        
        print(f"✂️  Created {len(chunks)} chunks ({duration_seconds/60:.1f} min total)")
        return chunks
    
    def transcribe_chunk(self, chunk_info):
        """
        Transcribe a single chunk
        
        Args:
            chunk_info: (chunk_file, start_time, end_time, chunk_index)
        
        Returns:
            Dictionary with transcription result
        """
        chunk_file, start_time, end_time, chunk_index = chunk_info
        
        try:
            if self.model_type == "faster-whisper":
                # Use faster-whisper
                segments, info = self.model.transcribe(str(chunk_file), beam_size=5)
                
                # Convert segments to list and adjust timestamps
                chunk_segments = []
                for segment in segments:
                    chunk_segments.append({
                        'start': segment.start + start_time,
                        'end': segment.end + start_time,
                        'text': segment.text.strip()
                    })
                
                result = {
                    'segments': chunk_segments,
                    'language': info.language,
                    'language_probability': info.language_probability
                }
            else:
                # Use standard whisper
                if WHISPER_AVAILABLE:
                    result = self.model.transcribe(str(chunk_file))
                    
                    # Adjust timestamps
                    for segment in result['segments']:
                        segment['start'] += start_time
                        segment['end'] += start_time
                else:
                    raise RuntimeError("No whisper implementation available")
            
            # Report progress
            with self.progress_lock:
                if self.progress_callback:
                    self.progress_callback(chunk_index, f"Completed chunk {chunk_index + 1}")
            
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'result': result,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error transcribing chunk {chunk_index}: {e}")
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'error': str(e),
                'success': False
            }
    
    def merge_chunk_results(self, chunk_results, total_duration):
        """
        Merge transcription results from all chunks
        
        Args:
            chunk_results: List of chunk transcription results
            total_duration: Total audio duration in seconds
        
        Returns:
            Combined transcription result
        """
        print("🔗 Merging chunk results...")
        
        # Sort by chunk index
        chunk_results.sort(key=lambda x: x['chunk_index'])
        
        all_segments = []
        languages = {}
        
        for chunk_result in chunk_results:
            if not chunk_result['success']:
                print(f"⚠️  Skipping failed chunk {chunk_result['chunk_index']}")
                continue
            
            result = chunk_result['result']
            
            # Collect language info
            lang = result.get('language', 'unknown')
            if lang in languages:
                languages[lang] += 1
            else:
                languages[lang] = 1
            
            # Add segments
            for segment in result.get('segments', []):
                all_segments.append(segment)
        
        # Remove overlapping segments (keep the one with better confidence or earlier chunk)
        all_segments.sort(key=lambda x: x['start'])
        
        # Simple overlap removal - keep first occurrence
        merged_segments = []
        last_end_time = 0
        
        for segment in all_segments:
            # If this segment starts before the last one ended, it's an overlap
            if segment['start'] < last_end_time:
                # Skip if it's mostly overlapping
                overlap_ratio = (last_end_time - segment['start']) / (segment['end'] - segment['start'])
                if overlap_ratio > 0.5:
                    continue
                # Otherwise, adjust start time
                segment['start'] = last_end_time
            
            merged_segments.append(segment)
            last_end_time = segment['end']
        
        # Determine most common language
        most_common_lang = max(languages.keys(), key=lambda k: languages[k]) if languages else 'unknown'
        
        # Build full text
        full_text = ' '.join(segment['text'].strip() for segment in merged_segments)
        
        return {
            'text': full_text,
            'segments': merged_segments,
            'language': most_common_lang,
            'language_probability': 0.9,  # Estimated
            'duration': total_duration
        }
    
    def transcribe_file_chunked(self, audio_path, progress_callback=None):
        """
        Transcribe a large audio file using chunked parallel processing
        
        Args:
            audio_path: Path to audio file
            progress_callback: Function to call with progress updates
        
        Returns:
            Combined transcription result
        """
        self.progress_callback = progress_callback
        
        audio_path = Path(audio_path)
        start_time = time.time()
        
        print(f"🎵 Starting chunked transcription: {audio_path.name}")
        
        # Get audio duration
        duration = self.get_audio_duration(audio_path)
        if duration is None:
            raise ValueError("Could not determine audio duration")
        
        print(f"⏱️  Audio duration: {duration/60:.1f} minutes")
        
        # Load model
        self.load_model()
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Split audio into chunks
            chunks = self.split_audio_into_chunks(audio_path, temp_path)
            
            if progress_callback:
                progress_callback(0, f"Created {len(chunks)} chunks, starting transcription...")
            
            # Prepare chunk info for parallel processing
            chunk_infos = [
                (chunk_file, start_time, end_time, i)
                for i, (chunk_file, start_time, end_time) in enumerate(chunks)
            ]
            
            # Process chunks in parallel
            print(f"🚀 Processing {len(chunks)} chunks with {self.max_workers} workers...")
            
            chunk_results = []
            completed_chunks = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all chunk transcription tasks
                future_to_chunk = {
                    executor.submit(self.transcribe_chunk, chunk_info): chunk_info
                    for chunk_info in chunk_infos
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_info = future_to_chunk[future]
                    try:
                        result = future.result()
                        chunk_results.append(result)
                        completed_chunks += 1
                        
                        progress_msg = f"Completed {completed_chunks}/{len(chunks)} chunks"
                        print(f"✅ {progress_msg}")
                        
                        if progress_callback:
                            progress_callback(completed_chunks, progress_msg)
                        
                    except Exception as e:
                        print(f"❌ Chunk failed: {e}")
                        chunk_results.append({
                            'chunk_index': chunk_info[3],
                            'start_time': chunk_info[1],
                            'end_time': chunk_info[2],
                            'error': str(e),
                            'success': False
                        })
        
        # Merge results
        if progress_callback:
            progress_callback(len(chunks), "Merging results...")
        
        final_result = self.merge_chunk_results(chunk_results, duration)
        
        # Add timing information
        processing_time = time.time() - start_time
        speed_ratio = duration / processing_time if processing_time > 0 else 0
        
        final_result.update({
            'processing_time': processing_time,
            'speed_ratio': speed_ratio,
            'chunks_processed': len(chunks),
            'chunks_succeeded': sum(1 for r in chunk_results if r['success'])
        })
        
        print(f"✅ Chunked transcription completed!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"⚡ Speed ratio: {speed_ratio:.1f}x real-time")
        print(f"📊 Chunks: {final_result['chunks_succeeded']}/{len(chunks)} succeeded")
        
        return final_result

def test_chunked_transcription():
    """Test the chunked transcription system"""
    
    # Find an audio file to test with
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(".").glob(f"*{ext}"))
    
    if not audio_files:
        print("❌ No audio files found for testing")
        return
    
    # Use the first (preferably large) file
    test_file = audio_files[0]
    
    print(f"🧪 Testing chunked transcription with: {test_file.name}")
    
    # Progress callback
    def progress_update(completed, message):
        print(f"📊 Progress: {message}")
    
    # Create transcriber
    transcriber = ChunkedWhisperTranscriber(
        model_size="large-v3",
        chunk_length_minutes=3,  # Smaller chunks for testing
        overlap_seconds=2,
        max_workers=2
    )
    
    try:
        # Transcribe
        result = transcriber.transcribe_file_chunked(test_file, progress_update)
        
        # Save result
        output_file = f"{test_file.stem}_chunked_transcription.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        print(f"💾 Transcription saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_chunked_transcription()
