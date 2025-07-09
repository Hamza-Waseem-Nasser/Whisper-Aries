#!/usr/bin/env python3
"""
Final Optimized Whisper Transcriber
Combines the best of parallel processing, model preloading, and memory optimization
Optimized for RTX 3050 4GB and similar systems
"""

import os
import sys
import json
import time
import threading
import queue
import gc
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkWork:
    """Work item for a chunk"""
    chunk_id: int
    chunk_audio: np.ndarray
    chunk_start: float
    chunk_duration: float
    transcribe_params: Dict

@dataclass
class ChunkResult:
    """Result from processing a chunk"""
    chunk_id: int
    segments: List[Dict]
    processing_time: float
    chunk_duration: float
    success: bool
    language: str = "unknown"
    language_probability: float = 0.0
    worker_id: str = "unknown"
    error: str = ""

class AudioPreloader:
    """Memory-based audio preloader for fast chunk access"""
    
    def __init__(self, audio_path: str, target_sr: int = 16000):
        self.audio_path = Path(audio_path)
        self.target_sr = target_sr
        self.audio_data = None
        self.duration = 0
        self.sample_rate = target_sr
        
        self._load_audio()
    
    def _load_audio(self):
        """Load entire audio file into memory once"""
        print(f"📂 Preloading audio from: {self.audio_path.name}")
        start_time = time.time()
        
        try:
            import soundfile as sf
            import librosa
            
            # Handle different file types
            file_ext = self.audio_path.suffix.lower()
            
            if file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.m4v', '.webm']:
                print(f"🎬 Detected video file ({file_ext}), extracting audio...")
                self.audio_data, self.sample_rate = librosa.load(
                    str(self.audio_path), 
                    sr=self.target_sr,
                    mono=True,
                    dtype=np.float32  # Critical: float32 for ONNXRuntime compatibility
                )
            else:
                print(f"🎵 Loading audio file ({file_ext})...")
                try:
                    # Try soundfile first
                    audio_data, sample_rate = sf.read(str(self.audio_path))
                    
                    # Convert to mono if stereo
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    # Ensure float32 dtype
                    self.audio_data = audio_data.astype(np.float32)
                    self.sample_rate = sample_rate
                    
                    # Resample if needed
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
            
            print(f"✅ Audio loaded: {self.duration/3600:.2f}h in {load_time:.1f}s")
            memory_mb = self.audio_data.nbytes / 1024 / 1024
            print(f"💾 Audio data size: {memory_mb:.1f}MB in RAM")
            
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            print(f"💡 Tip: Install librosa and soundfile: pip install librosa soundfile")
            raise
    
    def get_chunk(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Get audio chunk by time range (lightning fast memory access)"""
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        # Return float32 chunk for compatibility
        return self.audio_data[start_sample:end_sample].astype(np.float32)

class ModelManager:
    """Thread-safe model manager with per-worker preloading"""
    
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
    
    def get_model(self, model_config: Dict, worker_id: int = 0):
        """Get or create a model for a specific worker"""
        try:
            from faster_whisper import WhisperModel
            import torch
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Determine device
        device = model_config['device']
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # Create unique model key for this worker
        model_key = f"{model_config['model']}_{device}_{worker_id}"
        
        with self._lock:
            if model_key not in self._models:
                try:
                    print(f"🔧 Loading model for worker {worker_id} on {device}")
                    
                    # Filter out unsupported arguments
                    model_args = {
                        "device": device,
                        "compute_type": model_config["compute_type"],
                        "download_root": "./models"
                    }
                    
                    # Add CPU threads only for CPU device
                    if device == "cpu":
                        model_args["cpu_threads"] = 2
                    
                    model = WhisperModel(
                        model_config["model"],
                        **model_args
                    )
                    
                    self._models[model_key] = model
                    print(f"✅ Model loaded for worker {worker_id} on {device}")
                    
                    # Quick test
                    test_audio = np.random.random(8000).astype(np.float32)
                    segments, info = model.transcribe(test_audio, language="en", beam_size=1)
                    list(segments)  # Consume generator
                    print(f"✅ Model test successful for worker {worker_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to load model for worker {worker_id}: {e}")
                    raise
            
            return self._models[model_key]

class OptimizedParallelTranscriber:
    """High-performance parallel transcriber with memory-based audio processing"""
    
    def __init__(self, 
                 model_size: str = "base",  # Start with base for 4GB GPU
                 device: str = "auto",
                 compute_type: str = "int8",
                 chunk_length_minutes: float = 3.0,  # Smaller chunks for better parallelization
                 overlap_seconds: float = 5.0,
                 num_workers: Optional[int] = None):
        
        self.model_config = {
            "model": model_size,
            "device": device,
            "compute_type": compute_type
        }
        
        self.chunk_length_minutes = chunk_length_minutes
        self.overlap_seconds = overlap_seconds
        
        # Optimize worker count for RTX 3050 4GB
        if num_workers is None:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if gpu_memory <= 6:  # 4-6GB GPU like RTX 3050
                        # For limited VRAM, use 1-2 workers depending on model size
                        if model_size in ["large-v3", "large-v2", "large"]:
                            self.num_workers = 1  # Large models need more VRAM
                        else:
                            self.num_workers = 2  # Base/small models can handle 2
                    else:
                        self.num_workers = 2  # Higher-end GPU
                else:
                    # CPU fallback
                    cpu_cores = psutil.cpu_count()
                    self.num_workers = min(3, max(1, cpu_cores // 2))
            except ImportError:
                self.num_workers = 1
        else:
            self.num_workers = num_workers
        
        self.model_manager = ModelManager()
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.stop_event = threading.Event()
        
        print(f"🚀 Optimized Parallel Transcriber Configuration:")
        print(f"   🧠 Model: {model_size}")
        print(f"   🎮 Device: {device}")
        print(f"   👥 Workers: {self.num_workers}")
        print(f"   📏 Chunk length: {chunk_length_minutes} minutes")
        print(f"   🔄 Overlap: {overlap_seconds} seconds")
        print(f"   🔧 Compute type: {compute_type}")
    
    def _worker_thread(self, worker_id: int):
        """Worker thread that processes chunks with pre-loaded model"""
        try:
            # Pre-load model for this worker
            print(f"🔧 Worker {worker_id} loading model...")
            model = self.model_manager.get_model(self.model_config, worker_id)
            print(f"✅ Worker {worker_id} ready with pre-loaded model")
            
            # Process chunks
            while not self.stop_event.is_set():
                try:
                    # Get work from queue
                    work_item = self.work_queue.get(timeout=1.0)
                    
                    if work_item is None:  # Poison pill
                        break
                    
                    # Process the chunk
                    result = self._transcribe_chunk(model, work_item, worker_id)
                    self.result_queue.put(result)
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Worker {worker_id} error: {e}")
                    if 'work_item' in locals():
                        error_result = ChunkResult(
                            chunk_id=getattr(work_item, 'chunk_id', -1),
                            segments=[],
                            processing_time=0,
                            chunk_duration=0,
                            success=False,
                            error=str(e),
                            worker_id=f"Worker-{worker_id}"
                        )
                        self.result_queue.put(error_result)
                        self.work_queue.task_done()
            
            print(f"🛑 Worker {worker_id} stopped")
                    
        except Exception as e:
            print(f"❌ Worker {worker_id} failed to initialize: {e}")
    
    def _transcribe_chunk(self, model, work_item: ChunkWork, worker_id: int) -> ChunkResult:
        """Transcribe a single chunk using pre-loaded model"""
        try:
            start_time = time.time()
            
            # Ensure chunk audio is float32
            chunk_audio = work_item.chunk_audio.astype(np.float32)
            
            # Filter transcribe parameters to avoid API issues
            safe_params = {}
            allowed_params = {
                'language', 'beam_size', 'best_of', 'patience', 'length_penalty',
                'repetition_penalty', 'no_repeat_ngram_size', 'temperature',
                'compression_ratio_threshold', 'no_speech_threshold',
                'condition_on_previous_text', 'prompt_reset_on_temperature',
                'initial_prompt', 'prefix', 'suppress_blank', 'suppress_tokens',
                'without_timestamps', 'max_initial_timestamp', 'word_timestamps',
                'prepend_punctuations', 'append_punctuations', 'multilingual',
                'vad_filter', 'vad_parameters'
            }
            
            for key, value in work_item.transcribe_params.items():
                if key in allowed_params:
                    safe_params[key] = value
            
            # Transcribe chunk
            segments, info = model.transcribe(chunk_audio, **safe_params)
            
            # Process segments
            chunk_segments = []
            for segment in segments:
                adjusted_segment = {
                    "start": segment.start + work_item.chunk_start,
                    "end": segment.end + work_item.chunk_start,
                    "text": segment.text.strip(),
                    "avg_logprob": getattr(segment, 'avg_logprob', -1.0),
                    "no_speech_prob": getattr(segment, 'no_speech_prob', 0.0),
                    "chunk_id": work_item.chunk_id,
                    "worker_id": f"Worker-{worker_id}"
                }
                chunk_segments.append(adjusted_segment)
            
            processing_time = time.time() - start_time
            
            return ChunkResult(
                chunk_id=work_item.chunk_id,
                segments=chunk_segments,
                processing_time=processing_time,
                chunk_duration=work_item.chunk_duration,
                success=True,
                language=getattr(info, 'language', 'unknown'),
                language_probability=getattr(info, 'language_probability', 0.0),
                worker_id=f"Worker-{worker_id}"
            )
            
        except Exception as e:
            print(f"❌ Chunk {work_item.chunk_id} failed: {e}")
            return ChunkResult(
                chunk_id=work_item.chunk_id,
                segments=[],
                processing_time=0,
                chunk_duration=work_item.chunk_duration,
                success=False,
                error=str(e),
                worker_id=f"Worker-{worker_id}"
            )
    
    def start_workers(self):
        """Start worker threads with staggered initialization"""
        print(f"🚀 Starting {self.num_workers} worker threads...")
        
        for worker_id in range(self.num_workers):
            worker_thread = threading.Thread(
                target=self._worker_thread,
                args=(worker_id,),
                daemon=True,
                name=f"WhisperWorker-{worker_id}"
            )
            worker_thread.start()
            self.workers.append(worker_thread)
            
            # Stagger worker startup to avoid memory spikes
            if worker_id < self.num_workers - 1:
                time.sleep(1.5)  # Increased delay for stability
        
        # Wait for workers to initialize
        print("⏳ Waiting for workers to initialize...")
        time.sleep(3.0)
        print(f"✅ {self.num_workers} workers ready")
    
    def stop_workers(self):
        """Stop all worker threads"""
        print("🛑 Stopping workers...")
        self.stop_event.set()
        
        # Send poison pills
        for _ in range(self.num_workers):
            self.work_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        print("✅ All workers stopped")
    
    def transcribe_file(self, 
                       audio_path: str,
                       language: str = "auto",
                       output_formats: List[str] = ["txt", "json", "srt"]) -> Dict:
        """
        Transcribe an audio file using optimized parallel processing
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        print(f"🎵 Starting optimized transcription: {audio_path.name}")
        
        try:
            # Step 1: Preload audio into memory
            audio_preloader = AudioPreloader(str(audio_path))
            
            # Step 2: Calculate chunks
            chunk_length_seconds = self.chunk_length_minutes * 60
            total_chunks = int(np.ceil(audio_preloader.duration / chunk_length_seconds))
            
            print(f"📊 Audio duration: {audio_preloader.duration/60:.1f} minutes")
            print(f"📊 Creating {total_chunks} chunks of {self.chunk_length_minutes} minutes each")
            
            # Step 3: Start workers
            self.start_workers()
            
            # Step 4: Create and queue work items
            transcribe_params = {
                "language": None if language == "auto" else language,
                "beam_size": 1,  # Fast beam size for speed
                "best_of": 1,
                "temperature": 0,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.4,
                "vad_filter": True  # Voice activity detection for better quality
            }
            
            for chunk_id in range(total_chunks):
                start_sec = chunk_id * chunk_length_seconds
                end_sec = min(start_sec + chunk_length_seconds + self.overlap_seconds, 
                             audio_preloader.duration)
                
                chunk_audio = audio_preloader.get_chunk(start_sec, end_sec)
                chunk_duration = len(chunk_audio) / audio_preloader.sample_rate
                
                work_item = ChunkWork(
                    chunk_id=chunk_id,
                    chunk_audio=chunk_audio,
                    chunk_start=start_sec,
                    chunk_duration=chunk_duration,
                    transcribe_params=transcribe_params
                )
                
                self.work_queue.put(work_item)
            
            # Step 5: Collect results
            results = []
            completed_chunks = 0
            total_processing_time = 0
            
            print("🔄 Processing chunks...")
            
            while completed_chunks < total_chunks:
                try:
                    result = self.result_queue.get(timeout=120.0)  # Increased timeout for large chunks
                    results.append(result)
                    completed_chunks += 1
                    total_processing_time += result.processing_time
                    
                    if result.success:
                        progress = (completed_chunks / total_chunks) * 100
                        print(f"✅ Chunk {result.chunk_id+1}/{total_chunks} completed "
                              f"({result.processing_time:.1f}s, {len(result.segments)} segments) "
                              f"[{progress:.1f}%]")
                    else:
                        print(f"❌ Chunk {result.chunk_id+1}/{total_chunks} failed: {result.error}")
                    
                except queue.Empty:
                    print(f"⚠️ Timeout waiting for results. Completed: {completed_chunks}/{total_chunks}")
                    # Check if any workers are still alive
                    active_workers = sum(1 for w in self.workers if w.is_alive())
                    print(f"⚠️ Active workers: {active_workers}/{self.num_workers}")
                    if active_workers == 0:
                        print("⚠️ All workers stopped, breaking...")
                        break
            
            # Step 6: Stop workers
            self.stop_workers()
            
            # Step 7: Combine results
            results.sort(key=lambda x: x.chunk_id)
            all_segments = []
            
            for result in results:
                if result.success:
                    all_segments.extend(result.segments)
            
            # Remove overlapping segments
            all_segments = self._remove_overlaps(all_segments)
            
            total_time = time.time() - start_time
            real_time_factor = audio_preloader.duration / total_time
            
            print(f"🎉 Transcription completed!")
            print(f"⏱️ Total time: {total_time:.1f}s")
            print(f"🚀 Real-time factor: {real_time_factor:.1f}x")
            print(f"📝 Total segments: {len(all_segments)}")
            
            # Step 8: Generate outputs
            outputs = self._generate_outputs(audio_path, all_segments, output_formats)
            
            return {
                "success": True,
                "segments": all_segments,
                "processing_time": total_time,
                "real_time_factor": real_time_factor,
                "audio_duration": audio_preloader.duration,
                "total_chunks": total_chunks,
                "successful_chunks": sum(1 for r in results if r.success),
                "outputs": outputs
            }
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            self.stop_workers()
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _remove_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """Remove overlapping segments from chunk boundaries"""
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        cleaned_segments = [segments[0]]
        
        for segment in segments[1:]:
            last_segment = cleaned_segments[-1]
            
            # If segments overlap significantly, skip the later one
            if segment['start'] < last_segment['end'] - 1.0:  # 1 second overlap threshold
                continue
            
            cleaned_segments.append(segment)
        
        return cleaned_segments
    
    def _generate_outputs(self, audio_path: Path, segments: List[Dict], formats: List[str]) -> Dict:
        """Generate output files in requested formats"""
        outputs = {}
        base_name = audio_path.stem
        
        try:
            # Text format
            if "txt" in formats:
                txt_path = audio_path.parent / f"{base_name}_optimized_transcription.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for segment in segments:
                        f.write(f"{segment['text']}\n")
                outputs["txt"] = str(txt_path)
                print(f"💾 Text saved: {txt_path.name}")
            
            # JSON format
            if "json" in formats:
                json_path = audio_path.parent / f"{base_name}_optimized_transcription.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "transcription": segments,
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "audio_file": str(audio_path),
                            "total_segments": len(segments),
                            "model": self.model_config["model"],
                            "device": self.model_config["device"]
                        }
                    }, f, indent=2, ensure_ascii=False)
                outputs["json"] = str(json_path)
                print(f"💾 JSON saved: {json_path.name}")
            
            # SRT format
            if "srt" in formats:
                srt_path = audio_path.parent / f"{base_name}_optimized_transcription.srt"
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for i, segment in enumerate(segments, 1):
                        start_time = self._seconds_to_srt_time(segment['start'])
                        end_time = self._seconds_to_srt_time(segment['end'])
                        f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")
                outputs["srt"] = str(srt_path)
                print(f"💾 SRT saved: {srt_path.name}")
            
        except Exception as e:
            print(f"⚠️ Error saving outputs: {e}")
        
        return outputs
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final Optimized Whisper Transcriber")
    parser.add_argument("audio_file", help="Path to audio/video file")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                       help="Whisper model size (default: base for RTX 3050)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use (default: auto)")
    parser.add_argument("--language", default="auto", help="Language code (default: auto-detect)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: auto)")
    parser.add_argument("--chunk-minutes", type=float, default=3.0, help="Chunk length in minutes")
    parser.add_argument("--formats", nargs="+", default=["txt", "json", "srt"], 
                       choices=["txt", "json", "srt"], help="Output formats")
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return 1
    
    print("🚀 Final Optimized Whisper Transcriber")
    print("=" * 50)
    
    # Create transcriber
    transcriber = OptimizedParallelTranscriber(
        model_size=args.model,
        device=args.device,
        num_workers=args.workers,
        chunk_length_minutes=args.chunk_minutes
    )
    
    # Run transcription
    result = transcriber.transcribe_file(
        str(audio_path),
        language=args.language,
        output_formats=args.formats
    )
    
    if result["success"]:
        print("🎉 Transcription completed successfully!")
        return 0
    else:
        print(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
