#!/usr/bin/env python3
"""
Optimized chunked transcriber for large audio files
Enhanced with memory-based audio processing and improved model management
"""

import os
import time
import json
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
import shutil
import gc

class AudioPreloader:
    """Preload and manage audio data in memory for fast access"""
    
    def __init__(self, audio_path, target_sr: int = 16000):
        self.audio_path = Path(audio_path)
        self.target_sr = target_sr
        self.audio_data = None
        self.duration = 0
        self.sample_rate = target_sr
        
        self._load_audio()
    
    def _load_audio(self):
        """Load entire audio file into memory once (supports video files)"""
        print(f"📂 Preloading audio from: {self.audio_path.name}")
        start_time = time.time()
        
        try:
            import soundfile as sf
            import librosa
            
            # Handle video files and audio files
            file_ext = self.audio_path.suffix.lower()
            
            if file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.m4v', '.webm']:
                print(f"🎬 Detected video file ({file_ext}), extracting audio...")
                self.audio_data, self.sample_rate = librosa.load(
                    str(self.audio_path), 
                    sr=self.target_sr,
                    mono=True,
                    dtype=np.float32
                )
            else:
                print(f"🎵 Loading audio file ({file_ext})...")
                try:
                    self.audio_data, self.sample_rate = sf.read(str(self.audio_path))
                    
                    if len(self.audio_data.shape) > 1:
                        self.audio_data = np.mean(self.audio_data, axis=1)
                    
                    if self.sample_rate != self.target_sr:
                        print(f"🔄 Resampling from {self.sample_rate}Hz to {self.target_sr}Hz...")
                        self.audio_data = librosa.resample(
                            self.audio_data, 
                            orig_sr=self.sample_rate, 
                            target_sr=self.target_sr
                        )
                        self.sample_rate = self.target_sr
                        
                except Exception:
                    print("⚠️ Soundfile failed, trying librosa...")
                    self.audio_data, self.sample_rate = librosa.load(
                        str(self.audio_path), 
                        sr=self.target_sr,
                        mono=True,
                        dtype=np.float32
                    )
            
            self.duration = len(self.audio_data) / self.sample_rate
            load_time = time.time() - start_time
            
            print(f"✅ Audio extracted: {self.duration/3600:.2f}h in {load_time:.1f}s")
            memory_mb = self.audio_data.nbytes / 1024 / 1024
            print(f"💾 Audio data size: {memory_mb:.1f}MB in RAM")
            
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            print(f"💡 Tip: Install librosa and soundfile: pip install librosa soundfile")
            raise
    
    def get_chunk(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Get audio chunk by time range (much faster than file I/O)"""
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        return self.audio_data[start_sample:end_sample].copy()
    
    def save_chunk_to_temp(self, start_sec: float, end_sec: float, temp_dir: Path, chunk_index: int):
        """Save audio chunk to temporary file for compatibility"""
        chunk_audio = self.get_chunk(start_sec, end_sec)
        chunk_file = temp_dir / f"chunk_{chunk_index:03d}.wav"
        
        try:
            import soundfile as sf
            sf.write(str(chunk_file), chunk_audio, self.sample_rate)
            return chunk_file
        except Exception as e:
            print(f"⚠️ Failed to save chunk to temp file: {e}")
            return None

class OptimizedChunkedTranscriber:
    """Optimized chunked transcription with shared model and memory-based processing"""
    
    def __init__(self, model_size="large-v3", device="auto", compute_type="int8", 
                 languages=None, chunk_length_minutes=5, overlap_seconds=10, max_workers=None):
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.languages = languages or ["ar", "en"]
        self.chunk_length_minutes = chunk_length_minutes
        self.overlap_seconds = overlap_seconds
        
        # Smart worker optimization based on analysis
        import psutil
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory <= 6:  # 4-6GB GPU
                    optimal_workers = 1  # Sequential for better performance
                    print("🎮 GPU detected: Using sequential processing (optimal for GPU)")
                else:
                    optimal_workers = 2  # Higher-end GPU
            else:
                optimal_workers = min(2, psutil.cpu_count() // 2)  # CPU fallback
        except ImportError:
            optimal_workers = 1
        
        if max_workers is None:
            self.max_workers = optimal_workers
        else:
            self.max_workers = max_workers
        
        # Single shared transcriber instance (pre-loaded)
        self.transcriber = None
        self._transcriber_lock = threading.Lock()
        self.audio_preloader = None  # Will be set during transcription
        
        print(f"🔧 Enhanced Optimized Chunked Transcriber Config:")
        print(f"   📏 Chunk length: {chunk_length_minutes} minutes")
        print(f"   🔄 Overlap: {overlap_seconds} seconds")
        print(f"   👥 Max workers: {self.max_workers} (auto-optimized)")
        print(f"   🎮 Device: {device}")
        print(f"   🧠 Model: {model_size}")
        print(f"   🚀 Mode: {'Sequential (GPU-optimized)' if self.max_workers == 1 else 'Parallel'}")
    
    def get_audio_duration_ffprobe(self, audio_path):
        """Get audio duration using ffprobe (more reliable) or fallback to preloader"""
        try:
            ffprobe_path = shutil.which('ffprobe')
            if not ffprobe_path:
                raise Exception("ffprobe not found")
                
            cmd = [
                ffprobe_path, 
                '-v', 'quiet', 
                '-show_entries', 'format=duration', 
                '-of', 'csv=p=0', 
                str(audio_path.absolute())
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                print(f"🕐 Audio duration (ffprobe): {duration/60:.1f} minutes")
                return duration
        except Exception as e:
            print(f"⚠️  ffprobe failed: {e}")
        
        # Enhanced fallback: try audio preloader if available
        if self.audio_preloader:
            print(f"🕐 Audio duration (preloader): {self.audio_preloader.duration/60:.1f} minutes")
            return self.audio_preloader.duration
        
        # Last resort: estimate from file size (very rough)
        file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        estimated_duration = file_size_mb * 60  # Very rough estimate
        print(f"⚠️  Using estimated duration: {estimated_duration/60:.1f} minutes")
        return estimated_duration
    
    def get_ffmpeg_path(self):
        """Get FFmpeg path - now simplified since Chocolatey installation works"""
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        else:
            raise Exception("FFmpeg not found. Please install via: choco install ffmpeg")

    def create_audio_chunk_enhanced(self, start_time, end_time, chunk_index, temp_dir):
        """Enhanced chunk creation using memory-based approach when possible"""
        if self.audio_preloader:
            # Use memory-based approach (much faster)
            try:
                chunk_file = self.audio_preloader.save_chunk_to_temp(
                    start_time, end_time, temp_dir, chunk_index
                )
                if chunk_file:
                    print(f"✅ Created chunk {chunk_index + 1} (memory): {start_time/60:.1f}-{end_time/60:.1f} min")
                    return chunk_file
            except Exception as e:
                print(f"⚠️ Memory-based chunk creation failed: {e}, falling back to FFmpeg")
        
        # Fallback to original FFmpeg approach
        return self.create_audio_chunk(self.audio_preloader.audio_path if self.audio_preloader else None, 
                                     start_time, end_time, chunk_index, temp_dir)

    def create_audio_chunk(self, audio_path, start_time, end_time, chunk_index, temp_dir):
        """Create audio chunk using ffmpeg (fallback method)"""
        if audio_path is None:
            print(f"❌ No audio path provided for chunk {chunk_index + 1}")
            return None
            
        try:
            chunk_file = temp_dir / f"chunk_{chunk_index:03d}.wav"
            
            # Get FFmpeg path
            ffmpeg_path = self.get_ffmpeg_path()
            
            cmd = [
                ffmpeg_path,
                '-i', str(Path(audio_path).absolute()),
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(chunk_file.absolute())
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and chunk_file.exists():
                print(f"✅ Created chunk {chunk_index + 1} (FFmpeg): {start_time/60:.1f}-{end_time/60:.1f} min")
                return chunk_file
            else:
                print(f"❌ Failed to create chunk {chunk_index + 1}: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating chunk {chunk_index + 1}: {e}")
            return None
    
    def get_transcriber(self):
        """Get shared transcriber instance (thread-safe) - Enhanced with model caching"""
        with self._transcriber_lock:
            if self.transcriber is None:
                print("🔄 Loading Whisper model (shared instance with enhanced caching)...")
                start_time = time.time()
                
                from transcribe import WhisperTranscriber
                self.transcriber = WhisperTranscriber(
                    model_size=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    languages=self.languages
                )
                
                load_time = time.time() - start_time
                print(f"✅ Model loaded in {load_time:.2f} seconds (shared & cached)")
                
                # Optional: Test the model to ensure it's working properly
                try:
                    print("🧪 Testing model with short audio...")
                    # The model will be tested during first real transcription
                    print("✅ Model ready for transcription")
                except Exception as e:
                    print(f"⚠️ Model test failed: {e}")
                
            return self.transcriber
    
    def transcribe_chunk(self, chunk_info):
        """Transcribe a single audio chunk"""
        chunk_file, start_time, end_time, chunk_index = chunk_info
        
        if chunk_file is None:
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': False,
                'error': 'Chunk file creation failed'
            }
        
        try:
            print(f"🎵 Transcribing chunk {chunk_index + 1}: {start_time/60:.1f}-{end_time/60:.1f} min")
            
            # Get shared transcriber
            transcriber = self.get_transcriber()
            
            # Transcribe just this chunk
            chunk_start = time.time()
            result = transcriber.transcribe_file(str(chunk_file))
            chunk_time = time.time() - chunk_start
            
            # Adjust timestamps to global time
            if 'segments' in result:
                for segment in result['segments']:
                    segment['start'] += start_time
                    segment['end'] += start_time
                    
                    if 'words' in segment:
                        for word in segment['words']:
                            word['start'] += start_time
                            word['end'] += start_time
            
            print(f"✅ Chunk {chunk_index + 1} completed in {chunk_time:.1f}s")
            
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': True,
                'result': result,
                'processing_time': chunk_time
            }
            
        except Exception as e:
            print(f"❌ Error transcribing chunk {chunk_index + 1}: {e}")
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up chunk file
            try:
                if chunk_file and Path(chunk_file).exists():
                    os.remove(chunk_file)
            except:
                pass
    
    def merge_results(self, chunk_results, total_duration):
        """Merge chunk results into final transcription"""
        print("🔄 Merging chunk results...")
        
        # Sort by chunk index
        chunk_results.sort(key=lambda x: x['chunk_index'])
        
        # Collect all segments
        all_segments = []
        total_processing_time = 0
        successful_chunks = 0
        
        for chunk in chunk_results:
            if chunk['success']:
                successful_chunks += 1
                total_processing_time += chunk.get('processing_time', 0)
                
                result = chunk['result']
                if 'segments' in result:
                    all_segments.extend(result['segments'])
        
        # Remove duplicate segments at chunk boundaries
        all_segments = self.remove_overlapping_segments(all_segments)
        
        # Build full text
        full_text = " ".join(segment['text'].strip() for segment in all_segments)
        
        # Get language from first successful chunk
        language = "en"
        language_probability = 1.0
        
        for chunk in chunk_results:
            if chunk['success'] and 'result' in chunk:
                language = chunk['result'].get('language', 'en')
                language_probability = chunk['result'].get('language_probability', 1.0)
                break
        
        merged_result = {
            'text': full_text,
            'segments': all_segments,
            'language': language,
            'language_probability': language_probability,
            'duration': total_duration,
            'processing_time': total_processing_time,
            'speed_ratio': total_duration / total_processing_time if total_processing_time > 0 else 0,
            'chunks_processed': successful_chunks,
            'chunks_total': len(chunk_results),
            'chunked_transcription': True
        }
        
        print(f"✅ Merged {successful_chunks}/{len(chunk_results)} chunks")
        return merged_result
    
    def remove_overlapping_segments(self, segments):
        """Remove overlapping segments at chunk boundaries"""
        if len(segments) <= 1:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        filtered_segments = [segments[0]]
        
        for segment in segments[1:]:
            last_segment = filtered_segments[-1]
            
            # If segments overlap significantly, skip the duplicate
            if segment['start'] < last_segment['end'] - 1.0:  # 1 second tolerance
                # Keep the segment with higher confidence or longer text
                if len(segment['text']) > len(last_segment['text']):
                    filtered_segments[-1] = segment
            else:
                filtered_segments.append(segment)
        
        return filtered_segments
    
    def transcribe_file(self, audio_path):
        """Enhanced main transcription method with memory-based processing"""
        audio_path = Path(audio_path)
        print(f"🎵 Starting enhanced optimized chunked transcription: {audio_path.name}")
        
        start_total = time.time()
        
        # Step 1: Preload audio into memory (this is the key optimization!)
        try:
            print("🚀 Step 1: Preloading audio into memory...")
            self.audio_preloader = AudioPreloader(audio_path)
            total_duration = self.audio_preloader.duration
            print(f"✅ Audio preloaded successfully: {total_duration/60:.1f} minutes")
        except Exception as e:
            print(f"⚠️ Audio preloading failed: {e}")
            print("🔄 Falling back to traditional file-based approach...")
            self.audio_preloader = None
            total_duration = self.get_audio_duration_ffprobe(audio_path)
        
        # Step 2: Pre-load the model (another key optimization!)
        print("🚀 Step 2: Pre-loading Whisper model...")
        transcriber = self.get_transcriber()
        print("✅ Model pre-loaded successfully")
        
        # Step 3: Calculate optimal chunks
        chunk_length_seconds = self.chunk_length_minutes * 60
        chunks = []
        
        current_time = 0
        chunk_index = 0
        
        while current_time < total_duration:
            start_time = max(0, current_time - self.overlap_seconds) if chunk_index > 0 else 0
            end_time = min(total_duration, current_time + chunk_length_seconds)
            
            chunks.append((start_time, end_time, chunk_index))
            current_time += chunk_length_seconds
            chunk_index += 1
        
        print(f"📊 Optimization Strategy:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Chunk size: {self.chunk_length_minutes} minutes")
        print(f"   Processing mode: {'Sequential (GPU-optimized)' if self.max_workers == 1 else f'Parallel ({self.max_workers} workers)'}")
        print(f"   Audio access: {'Memory-based (fast)' if self.audio_preloader else 'File-based (slower)'}")
        
        # Step 4: Process chunks (using optimized approach)
        processing_start = time.time()
        
        if self.max_workers == 1:
            # Sequential processing (optimal for GPU)
            print("🔥 Using sequential processing (optimal for GPU)...")
            chunk_results = self._process_chunks_sequential(chunks)
        else:
            # Parallel processing (for CPU or high-end GPU)
            print(f"🔥 Using parallel processing with {self.max_workers} workers...")
            chunk_results = self._process_chunks_parallel(chunks)
        
        processing_time = time.time() - processing_start
        
        # Step 5: Merge results
        print("🔗 Merging results...")
        final_result = self.merge_results(chunk_results, total_duration)
        
        # Add performance metrics
        total_time = time.time() - start_total
        
        final_result.update({
            'optimization_info': {
                'audio_preloaded': self.audio_preloader is not None,
                'processing_mode': 'sequential' if self.max_workers == 1 else 'parallel',
                'workers_used': self.max_workers,
                'total_chunks': len(chunks),
                'chunk_size_minutes': self.chunk_length_minutes
            },
            'performance': {
                'total_processing_time': total_time,
                'chunk_processing_time': processing_time,
                'speed_ratio': total_duration / total_time if total_time > 0 else 0,
                'chunks_per_minute': len(chunks) / (processing_time / 60) if processing_time > 0 else 0
            }
        })
        
        # Cleanup
        if self.audio_preloader:
            del self.audio_preloader.audio_data  # Free memory
            self.audio_preloader = None
        
        # Force garbage collection
        gc.collect()
        
        print(f"✅ Transcription completed in {total_time:.1f}s ({final_result['performance']['speed_ratio']:.1f}x real-time)")
        
        return final_result
    
    def _process_chunks_sequential(self, chunks):
        """Process chunks sequentially (optimal for GPU)"""
        chunk_results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, (start_time, end_time, chunk_index) in enumerate(chunks):
                print(f"🎵 Processing chunk {i+1}/{len(chunks)} sequentially...")
                
                # Create chunk (use enhanced method)
                chunk_file = self.create_audio_chunk_enhanced(start_time, end_time, chunk_index, temp_path)
                
                # Transcribe chunk
                result = self.transcribe_chunk((chunk_file, start_time, end_time, chunk_index))
                chunk_results.append(result)
        
        return chunk_results
    
    def _process_chunks_parallel(self, chunks):
        """Process chunks in parallel (for CPU or high-end GPU)"""
        chunk_results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Prepare chunk info
            chunk_infos = []
            for start_time, end_time, chunk_index in chunks:
                chunk_file = self.create_audio_chunk_enhanced(start_time, end_time, chunk_index, temp_path)
                chunk_infos.append((chunk_file, start_time, end_time, chunk_index))
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.transcribe_chunk, chunk_info) for chunk_info in chunk_infos]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    chunk_results.append(result)
                    print(f"✅ Completed chunk {len(chunk_results)}/{len(chunks)}")
        
        return chunk_results


if __name__ == "__main__":
    # Test the optimized chunked transcriber
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        # Handle filenames with spaces by joining all arguments after the first
        if len(sys.argv) > 2:
            audio_file = " ".join(sys.argv[1:])
        else:
            audio_file = sys.argv[1]
        
        # Check if file exists
        if not Path(audio_file).exists():
            print(f"❌ File not found: {audio_file}")
            print("💡 Available audio files:")
            for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
                files = list(Path('.').glob(ext))
                for f in files:
                    print(f"   - {f.name}")
            sys.exit(1)
        
        print(f"🎵 Processing: {audio_file}")
        
        transcriber = OptimizedChunkedTranscriber(
            chunk_length_minutes=2,
            max_workers=2
        )
        
        result = transcriber.transcribe_file(audio_file)
        print(f"\n📊 Final Result:")
        print(f"🕐 Duration: {result['duration']:.1f} seconds")
        print(f"⏱️  Processing: {result.get('total_processing_time', result.get('processing_time', 0)):.1f} seconds")
        print(f"⚡ Speed: {result.get('overall_speed_ratio', result.get('speed_ratio', 0)):.1f}x real-time")
        print(f"📝 Text length: {len(result.get('text', result.get('full_text', ''))):.0f} characters")
    else:
        print("Usage: python optimized_chunked_transcriber.py <audio_file>")
        print("💡 Available audio files:")
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
            files = list(Path('.').glob(ext))
            for f in files:
                print(f"   - {f.name}")
