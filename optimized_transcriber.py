#!/usr/bin/env python3
"""
Optimized Fast Whisper Transcriber
Tailored for RTX 3050 4GB and maximum speed
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
from typing import List, Dict, Optional, Union, Callable, Tuple
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

class OptimizedModelManager:
    """Optimized model manager for 4GB RTX 3050"""
    
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_config: Dict):
        """Get or create the model (single shared instance for memory efficiency)"""
        try:
            from faster_whisper import WhisperModel
            import torch
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        with self._lock:
            if self._model is None:
                print(f"🔧 Loading optimized model for RTX 3050...")
                
                # Optimized settings for 4GB VRAM
                self._model = WhisperModel(
                    model_config["model"],
                    device=model_config["device"],
                    compute_type=model_config["compute_type"],
                    download_root="./models",
                    num_workers=1,  # Single worker for GPU memory efficiency
                    cpu_threads=0,  # Let GPU handle everything
                )
                
                print(f"✅ Optimized model loaded")
                
                # Test the model
                test_audio = np.random.random(8000).astype(np.float32)
                segments, info = self._model.transcribe(test_audio, language="en", beam_size=1)
                list(segments)  # Consume generator
                print(f"✅ Model test successful")
            
            return self._model
    
    def cleanup_models(self):
        """Cleanup model"""
        with self._lock:
            if self._model is not None:
                try:
                    del self._model
                    self._model = None
                except Exception as e:
                    print(f"⚠️ Error cleaning up model: {e}")
            
        # GPU cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        gc.collect()
        print("🧹 Model cleanup completed")

class OptimizedWorkerPool:
    """Optimized worker pool for RTX 3050"""
    
    def __init__(self, model_config: Dict, num_workers: int):
        self.model_config = model_config
        self.num_workers = min(num_workers, 1)  # Force single worker for RTX 3050
        self.model_manager = OptimizedModelManager()
        
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.stop_event = threading.Event()
        
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize optimized worker"""
        print(f"🚀 Initializing {self.num_workers} optimized worker...")
        
        def worker_startup(worker_id):
            """Startup function for worker"""
            try:
                # Pre-load shared model
                model = self.model_manager.get_model(self.model_config)
                print(f"✅ Worker {worker_id} ready with shared model")
                
                # Main worker loop
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
                        
            except Exception as e:
                print(f"❌ Worker {worker_id} failed to initialize: {e}")
        
        # Start single worker
        worker_thread = threading.Thread(
            target=worker_startup,
            args=(0,),
            daemon=True,
            name="OptimizedWhisperWorker"
        )
        worker_thread.start()
        self.workers.append(worker_thread)
        
        # Wait for worker to be ready
        print("⏳ Waiting for worker to initialize...")
        time.sleep(2.0)
        print(f"✅ Optimized worker ready")
    
    def _transcribe_chunk(self, model, work_item: ChunkWork, worker_id: int) -> ChunkResult:
        """Transcribe a single chunk using optimized settings"""
        try:
            start_time = time.time()
            
            # Optimized transcription with aggressive settings for speed
            segments, info = model.transcribe(
                work_item.chunk_audio, 
                **work_item.transcribe_params
            )
            
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
            return ChunkResult(
                chunk_id=work_item.chunk_id,
                segments=[],
                processing_time=0,
                chunk_duration=0,
                success=False,
                error=str(e),
                worker_id=f"Worker-{worker_id}"
            )
    
    def add_work(self, work_item: ChunkWork):
        """Add work to the queue"""
        self.work_queue.put(work_item)
    
    def get_result(self, timeout: float = None) -> ChunkResult:
        """Get a result from the queue"""
        return self.result_queue.get(timeout=timeout)
    
    def stop(self):
        """Stop worker"""
        print("🛑 Stopping worker...")
        self.stop_event.set()
        
        # Add poison pill
        self.work_queue.put(None)
        
        # Wait for worker to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

class AudioPreloader:
    """Preload and manage audio data in memory for fast access"""
    
    def __init__(self, audio_path: Union[str, Path], target_sr: int = 16000):
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
            
            file_ext = self.audio_path.suffix.lower()
            print(f"🎵 Loading audio file ({file_ext})...")
            
            try:
                self.audio_data, self.sample_rate = sf.read(str(self.audio_path), dtype='float32')
                
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
            raise
    
    def get_chunk(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Get audio chunk by time range"""
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        # Ensure float32 data type for ONNXRuntime compatibility
        chunk = self.audio_data[start_sample:end_sample].copy()
        return chunk.astype(np.float32) if chunk.dtype != np.float32 else chunk

class OptimizedTranscriber:
    """Optimized transcriber for RTX 3050 4GB"""
    
    def __init__(self, config_file: str = "optimized_whisper_config.json"):
        self.load_config(config_file)
        self.model_manager = OptimizedModelManager()
        
    def load_config(self, config_file: str):
        """Load optimized configuration"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"🔧 Configuration loaded from {config_file}")
        except FileNotFoundError:
            print("⚠️ Config file not found, using optimized defaults")
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Use smaller, faster model for RTX 3050
                compute_type = "int8" if device == "cuda" else "int8"  # More aggressive quantization
            except ImportError:
                device = "cpu"
                compute_type = "int8"
                
            self.config = {
                "model": "base",  # Much smaller and faster than large-v3
                "device": device,
                "compute_type": compute_type
            }
            
            # Save optimized config
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"💾 Optimized config saved to {config_file}")
    
    def transcribe_optimized(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        chunk_size: int = 30,   # Smaller chunks for faster processing
        overlap: int = 2,       # Minimal overlap
        **transcribe_kwargs
    ) -> Dict:
        """
        Optimized transcription for RTX 3050
        """
        print(f"🚀 Optimized Transcription for RTX 3050: {audio_path}")
        print("🔧 Using aggressive optimization for maximum speed")
        print("=" * 60)
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Check dependencies
            import torch
            from faster_whisper import WhisperModel
            from tqdm.auto import tqdm
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Step 1: Preload audio
        audio_preloader = AudioPreloader(audio_path)
        audio_duration = audio_preloader.duration
        
        # Step 2: Calculate optimal chunking strategy for RTX 3050
        num_chunks = max(1, int(np.ceil(audio_duration / chunk_size)))
        
        print(f"📊 Optimized Strategy for RTX 3050:")
        print(f"   Audio Duration: {audio_duration/3600:.2f} hours")
        print(f"   Chunk Size: {chunk_size}s (optimized for 4GB VRAM)")
        print(f"   Total Chunks: {num_chunks}")
        print(f"   Model: {self.config['model']} (faster than large-v3)")
        print(f"   Compute Type: {self.config['compute_type']} (aggressive quantization)")
        
        # Step 3: Create optimized worker pool
        print(f"\\n🔥 Creating optimized worker pool...")
        worker_pool = OptimizedWorkerPool(self.config, 1)
        
        # Step 4: Prepare aggressive transcription parameters for speed
        transcribe_params = {
            'language': language,
            'beam_size': 1,  # Fastest beam size
            'temperature': 0.0,  # Deterministic, faster
            'condition_on_previous_text': False,  # Faster without context
            'vad_filter': True,  # Remove silence for faster processing
            'vad_parameters': {
                'min_silence_duration_ms': 1000,  # More aggressive silence removal
                'speech_pad_ms': 100,  # Less padding for speed
            },
            'no_speech_threshold': 0.6,  # Skip likely silence
            'compression_ratio_threshold': 2.4,  # Skip likely noise
            'logprob_threshold': -1.0,  # Less strict for speed
            'initial_prompt': None,  # No prompt for speed
        }
        
        # Step 5: Submit all work to the pool
        print(f"📦 Submitting {num_chunks} chunks to optimized worker...")
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size + overlap, audio_duration)
            chunk_duration = chunk_end - chunk_start
            
            # Extract chunk audio from preloaded data
            chunk_audio = audio_preloader.get_chunk(chunk_start, chunk_end)
            
            work_item = ChunkWork(
                chunk_id=chunk_idx,
                chunk_audio=chunk_audio,
                chunk_start=chunk_start,
                chunk_duration=chunk_duration,
                transcribe_params=transcribe_params
            )
            
            worker_pool.add_work(work_item)
        
        # Step 6: Collect results with progress tracking
        print(f"\\n⚡ Processing with optimized worker...")
        
        all_results = {}
        start_time = time.time()
        
        with tqdm(total=num_chunks, desc="Optimized processing") as pbar:
            for _ in range(num_chunks):
                try:
                    result = worker_pool.get_result(timeout=120)  # 2 min timeout per chunk
                    
                    if result.success:
                        all_results[result.chunk_id] = result
                        
                        # Update progress
                        speed = result.chunk_duration / result.processing_time if result.processing_time > 0 else 0
                        
                        pbar.set_postfix({
                            'Speed': f'{speed:.1f}x',
                            'Time': f'{result.processing_time:.1f}s'
                        })
                        pbar.update(1)
                    
                    else:
                        error_msg = result.error or 'Unknown error'
                        print(f"\\n❌ Chunk {result.chunk_id} failed: {error_msg}")
                        pbar.update(1)
                        
                except queue.Empty:
                    print(f"\\n⚠️ Timeout waiting for chunk result")
                    break
                except Exception as e:
                    print(f"\\n❌ Error collecting result: {e}")
                    break
        
        total_processing_time = time.time() - start_time
        
        # Step 7: Cleanup worker pool
        worker_pool.stop()
        
        # Step 8: Process and merge results
        print(f"\\n🔗 Merging results from {len(all_results)} chunks...")
        
        all_segments = []
        detected_language = "unknown"
        language_confidence = 0
        
        for chunk_id in sorted(all_results.keys()):
            result = all_results[chunk_id]
            all_segments.extend(result.segments)
            
            # Use language from first successful chunk
            if detected_language == "unknown" and result.language:
                detected_language = result.language
                language_confidence = result.language_probability
        
        # Remove overlaps
        merged_segments = self._merge_overlapping_segments(all_segments, overlap)
        
        # Generate full text
        full_text = " ".join([seg["text"].strip() for seg in merged_segments])
        
        # Calculate performance metrics
        total_audio_processed = sum(result.chunk_duration for result in all_results.values())
        average_speed = total_audio_processed / total_processing_time if total_processing_time > 0 else 0
        
        # Prepare results
        results = {
            "file": str(audio_path),
            "language": detected_language,
            "language_probability": language_confidence,
            "duration": audio_duration,
            "full_text": full_text,
            "segments": merged_segments,
            "optimization_info": {
                "model": self.config["model"],
                "compute_type": self.config["compute_type"],
                "chunk_size": chunk_size,
                "chunks_processed": len(all_results),
                "chunks_failed": num_chunks - len(all_results),
                "optimized_for": "RTX 3050 4GB"
            },
            "performance": {
                "total_processing_time": total_processing_time,
                "real_time_factor": average_speed,
                "model_efficiency": "high (base model)",
                "memory_efficiency": "optimized for 4GB VRAM"
            },
            "processing_date": datetime.now().isoformat(),
            "model_config": self.config
        }
        
        # Print performance summary
        self._print_summary(results)
        
        return results
    
    def _merge_overlapping_segments(self, segments: List[Dict], overlap: int) -> List[Dict]:
        """Merge overlapping segments"""
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            if next_seg['start'] <= current['end'] - overlap/2:  # Handle overlap
                # Merge segments
                if next_seg['end'] > current['end']:
                    current['text'] += " " + next_seg['text']
                    current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg.copy()
        
        merged.append(current)
        return merged
    
    def _print_summary(self, results: Dict):
        """Print comprehensive processing summary"""
        print("\\n" + "="*70)
        print("🚀 OPTIMIZED TRANSCRIPTION COMPLETED!")
        print("="*70)
        
        duration_hours = results['duration'] / 3600
        processing_mins = results['performance']['total_processing_time'] / 60
        speed_factor = results['performance']['real_time_factor']
        optimization_info = results['optimization_info']
        
        print(f"📹 Audio Duration: {duration_hours:.2f} hours ({results['duration']:.0f}s)")
        print(f"⚡ Processing Time: {processing_mins:.1f} minutes")
        print(f"🚀 Overall Speed: {speed_factor:.1f}x real-time")
        
        print(f"\\n🔧 Optimization Details:")
        print(f"   Model: {optimization_info['model']} (faster than large-v3)")
        print(f"   Compute Type: {optimization_info['compute_type']}")
        print(f"   Optimized For: {optimization_info['optimized_for']}")
        print(f"   Chunks Processed: {optimization_info['chunks_processed']}")
        
        # Performance assessment
        print(f"\\n📊 Performance Assessment:")
        if speed_factor >= 5:
            print("🏆 EXCELLENT! 5x+ real-time achieved!")
        elif speed_factor >= 3:
            print("✅ GREAT! 3x+ real-time achieved!")
        elif speed_factor >= 2:
            print("✅ GOOD! 2x+ real-time achieved!")
        else:
            print("⚠️ Performance below expectations")
            print("💡 Consider using smaller model or reducing chunk size")
        
        print("="*70)
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up resources...")
        self.model_manager.cleanup_models()

# Main execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        # Handle filenames with spaces
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
        
        # Create optimized transcriber
        transcriber = OptimizedTranscriber()
        
        try:
            result = transcriber.transcribe_optimized(
                audio_file,
                language="en",
                chunk_size=30,  # Optimized for RTX 3050
                overlap=2,      # Minimal overlap
            )
            
            # Save results
            output_dir = Path("transcriptions")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_name = Path(audio_file).stem
            
            # Text file
            txt_file = output_dir / f"{audio_name}_optimized_{timestamp}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("OPTIMIZED TRANSCRIPTION REPORT (RTX 3050)\\n")
                f.write("="*50 + "\\n")
                f.write(f"File: {audio_file}\\n")
                f.write(f"Processing Speed: {result['performance']['real_time_factor']:.1f}x real-time\\n")
                f.write(f"Model: {result['optimization_info']['model']}\\n")
                f.write(f"Optimized For: {result['optimization_info']['optimized_for']}\\n")
                f.write("\\n" + result['full_text'])
            
            # JSON report
            json_file = output_dir / f"{audio_name}_optimized_report_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\\n💾 Results saved:")
            print(f"   📝 Text: {txt_file}")
            print(f"   📊 Report: {json_file}")
            
            # Final results
            print(f"\\n📊 Final Result:")
            print(f"🕐 Duration: {result['duration']:.1f} seconds")
            print(f"⏱️  Processing: {result['performance']['total_processing_time']:.1f} seconds")
            print(f"⚡ Speed: {result['performance']['real_time_factor']:.1f}x real-time")
            print(f"📝 Text length: {len(result['full_text']):.0f} characters")
            print(f"🔧 Model: {result['optimization_info']['model']}")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            transcriber.cleanup()
            
    else:
        print("Usage: python optimized_transcriber.py <audio_file>")
        print("💡 Available audio files:")
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
            files = list(Path('.').glob(ext))
            for f in files:
                print(f"   - {f.name}")
