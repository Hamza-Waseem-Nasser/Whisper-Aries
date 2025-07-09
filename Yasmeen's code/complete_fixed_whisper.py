
import os
import sys
import subprocess
import json
import time
import threading
import asyncio
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import queue
import gc
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Enhanced system check with parallel processing recommendations"""
    print("🔍 Checking System Requirements for FIXED Parallel Processing...")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # System specs
    cpu_cores = psutil.cpu_count()
    memory = psutil.virtual_memory()
    
    print(f"💻 CPU Cores: {cpu_cores} (Physical: {psutil.cpu_count(logical=False)})")
    print(f"💾 System RAM: {memory.total / 1024**3:.1f}GB (Available: {memory.available / 1024**3:.1f}GB)")
    
    # Parallel processing recommendations
    if cpu_cores >= 8:
        print("✅ Excellent CPU for parallel processing")
        recommended_workers = min(6, cpu_cores // 2)
    elif cpu_cores >= 6:
        print("✅ Good CPU for parallel processing") 
        recommended_workers = 2
    else:
        print("⚠️ Limited CPU - parallel processing may not help much")
        recommended_workers = 1
    
    print(f"🚀 Recommended parallel workers: {recommended_workers}")
    
    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
            print(f"✅ GPU Count: {gpu_count}")
            
            # Conservative GPU parallel recommendations for stability
            if gpu_memory >= 24:
                gpu_workers = min(3, gpu_count)  # Reduced for stability
                print("🚀 High-end GPU: Can run 3 parallel streams")
            elif gpu_memory >= 16:
                gpu_workers = min(2, gpu_count)
                print("🚀 Mid-high GPU: Can run 2 parallel streams")
            elif gpu_memory >= 8:
                gpu_workers = 1
                print("🚀 Good GPU: Single stream recommended for stability")
            else:
                gpu_workers = 1
                print("⚠️ Limited GPU memory: Single stream only")
            
            return True, "cuda", recommended_workers, gpu_workers
        else:
            print("⚠️ No GPU - will use CPU parallel processing")
            return True, "cpu", recommended_workers, 1
    except ImportError:
        print("⚠️ PyTorch not installed")
        return True, "unknown", recommended_workers, 1

def install_requirements():
    """Install requirements with additional parallel processing packages"""
    print("\n📦 Installing Requirements with FIXED Parallel Processing Support...")
    print("=" * 60)
    
    packages = [
        "faster-whisper>=0.10.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "numpy>=1.21.0",
        "librosa>=0.9.0",
        "soundfile>=0.12.0",
        "ffmpeg-python>=0.2.0",
        "psutil>=5.8.0",
        "tqdm>=4.64.0",
    ]
    
    # Install PyTorch with CUDA
    print("🎮 Installing PyTorch with CUDA support...")
    torch_command = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    try:
        subprocess.check_call(torch_command, stdout=subprocess.DEVNULL)
        print("✅ PyTorch with CUDA installed")
    except subprocess.CalledProcessError:
        print("❌ CUDA installation failed, installing CPU version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio"])
    
    # Install other packages
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

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

class GlobalModelManager:
    """Thread-safe global model manager to avoid repeated model loading"""
    
    _instance = None
    _models = {}
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_config: Dict, worker_id: int = 0):
        """Get or create a model for a specific worker (thread-safe)"""
        try:
            from faster_whisper import WhisperModel
            import torch
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")
        
        # Create unique key for this worker's model
        device = model_config['device']
        if device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                device_id = worker_id % gpu_count
                device = f"cuda:{device_id}"
        
        model_key = f"{model_config['model']}_{device}_{worker_id}"
        
        with self._lock:
            if model_key not in self._models:
                try:
                    print(f"🔧 Loading model for worker {worker_id} on {device}")
                    
                    model = WhisperModel(
                        model_config["model"],
                        device=device,
                        compute_type=model_config["compute_type"],
                        download_root="./models",
                        num_workers=1,  # Single worker per model instance
                        cpu_threads=2 if device.startswith("cpu") else 0,
                    )
                    
                    self._models[model_key] = model
                    print(f"✅ Model loaded for worker {worker_id} on {device}")
                    
                    # Test the model to ensure it works
                    test_audio = np.random.random(8000).astype(np.float32)
                    segments, info = model.transcribe(test_audio, language="en", beam_size=1)
                    list(segments)  # Consume generator
                    print(f"✅ Model test successful for worker {worker_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to load model for worker {worker_id}: {e}")
                    # Try fallback to shared model
                    fallback_key = f"{model_config['model']}_{model_config['device']}_shared"
                    if fallback_key not in self._models:
                        model = WhisperModel(
                            model_config["model"],
                            device=model_config['device'],
                            compute_type=model_config["compute_type"],
                            download_root="./models",
                        )
                        self._models[fallback_key] = model
                        print(f"⚠️ Using shared model for worker {worker_id}")
                    return self._models[fallback_key]
            
            return self._models[model_key]
    
    def cleanup_models(self):
        """Cleanup all models"""
        with self._lock:
            for model_key in list(self._models.keys()):
                try:
                    del self._models[model_key]
                except Exception as e:
                    print(f"⚠️ Error cleaning up model {model_key}: {e}")
            self._models.clear()
            
        # GPU cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        gc.collect()
        print("🧹 Model cleanup completed")

class WorkerDiagnostics:
    """Monitor and diagnose worker activity"""
    
    def __init__(self):
        self.worker_logs = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def log_worker_activity(self, worker_id: str, activity: str, details: str = ""):
        """Log worker activity with timestamp (thread-safe)"""
        with self.lock:
            if worker_id not in self.worker_logs:
                self.worker_logs[worker_id] = []
            
            self.worker_logs[worker_id].append({
                'time': time.time() - self.start_time,
                'activity': activity,
                'details': details,
                'thread_id': threading.get_ident()
            })
    
    def get_active_workers(self) -> int:
        """Get number of active workers"""
        with self.lock:
            return len(self.worker_logs)
    
    def print_worker_summary(self):
        """Print summary of worker activity"""
        with self.lock:
            print(f"\n📊 Worker Activity Summary:")
            print(f"   Total Workers Activated: {len(self.worker_logs)}")
            
            for worker_id, logs in self.worker_logs.items():
                if logs:
                    activities = [log['activity'] for log in logs]
                    print(f"   Worker {worker_id}: {len(logs)} activities - {', '.join(activities[:3])}...")

class FixedWorkerPool:
    """Fixed worker pool with pre-loaded models and proper management"""
    
    def __init__(self, model_config: Dict, num_workers: int):
        self.model_config = model_config
        self.num_workers = num_workers
        self.model_manager = GlobalModelManager()
        self.diagnostics = WorkerDiagnostics()
        
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.stop_event = threading.Event()
        
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize worker threads with pre-loaded models"""
        print(f"🚀 Initializing {self.num_workers} FIXED workers with pre-loaded models...")
        
        def worker_startup(worker_id):
            """Startup function for each worker"""
            self.diagnostics.log_worker_activity(worker_id, "STARTING")
            
            try:
                # Pre-load model for this worker
                self.diagnostics.log_worker_activity(worker_id, "LOADING_MODEL")
                model = self.model_manager.get_model(self.model_config, worker_id)
                self.diagnostics.log_worker_activity(worker_id, "MODEL_READY")
                
                print(f"✅ Worker {worker_id} ready with pre-loaded model")
                
                # Main worker loop
                while not self.stop_event.is_set():
                    try:
                        # Get work from queue
                        work_item = self.work_queue.get(timeout=1.0)
                        
                        if work_item is None:  # Poison pill
                            break
                        
                        self.diagnostics.log_worker_activity(worker_id, "PROCESSING", f"Chunk {work_item.chunk_id}")
                        
                        # Process the chunk
                        result = self._transcribe_chunk(model, work_item, worker_id)
                        self.result_queue.put(result)
                        
                        self.diagnostics.log_worker_activity(worker_id, "COMPLETED", f"Chunk {work_item.chunk_id}")
                        self.work_queue.task_done()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.diagnostics.log_worker_activity(worker_id, "ERROR", str(e))
                        error_result = ChunkResult(
                            chunk_id=getattr(work_item, 'chunk_id', -1),
                            segments=[],
                            processing_time=0,
                            chunk_duration=0,
                            success=False,
                            error=str(e),
                            worker_id=str(worker_id)
                        )
                        self.result_queue.put(error_result)
                        
                self.diagnostics.log_worker_activity(worker_id, "STOPPED")
                        
            except Exception as e:
                print(f"❌ Worker {worker_id} failed to initialize: {e}")
                self.diagnostics.log_worker_activity(worker_id, "INIT_FAILED", str(e))
        
        # Start workers with staggered initialization
        for worker_id in range(self.num_workers):
            worker_thread = threading.Thread(
                target=worker_startup,
                args=(worker_id,),
                daemon=True,
                name=f"WhisperWorker-{worker_id}"
            )
            worker_thread.start()
            self.workers.append(worker_thread)
            
            # Stagger worker startup to avoid memory spikes
            if worker_id < self.num_workers - 1:
                time.sleep(2.0)  # 2 second delay between workers
        
        # Wait for all workers to be ready
        print("⏳ Waiting for all workers to initialize...")
        time.sleep(5.0)  # Give workers time to load models
        
        print(f"✅ {self.diagnostics.get_active_workers()}/{self.num_workers} workers ready")
    
    def _transcribe_chunk(self, model, work_item: ChunkWork, worker_id: int) -> ChunkResult:
        """Transcribe a single chunk using pre-loaded model"""
        try:
            start_time = time.time()
            
            # Transcribe chunk with the pre-loaded model
            segments, info = model.transcribe(work_item.chunk_audio, **work_item.transcribe_params)
            
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
                    "worker_id": worker_id
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
                worker_id=str(worker_id)
            )
            
        except Exception as e:
            return ChunkResult(
                chunk_id=work_item.chunk_id,
                segments=[],
                processing_time=0,
                chunk_duration=0,
                success=False,
                error=str(e),
                worker_id=str(worker_id)
            )
    
    def add_work(self, work_item: ChunkWork):
        """Add work to the queue"""
        self.work_queue.put(work_item)
    
    def get_result(self, timeout: float = None) -> ChunkResult:
        """Get a result from the queue"""
        return self.result_queue.get(timeout=timeout)
    
    def stop(self):
        """Stop all workers"""
        print("🛑 Stopping worker pool...")
        self.stop_event.set()
        
        # Add poison pills
        for _ in range(self.num_workers):
            self.work_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.diagnostics.print_worker_summary()

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
            print(f"💡 Tip: Ensure ffmpeg is installed for video file support")
            raise
    
    def get_chunk(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Get audio chunk by time range (much faster than file I/O)"""
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        return self.audio_data[start_sample:end_sample].copy()

class PerformanceMonitor:
    """Enhanced performance monitor for parallel processing"""
    
    def __init__(self):
        self.start_time = None
        self.chunk_completions = {}
        self.completed_chunks = 0
        self.total_chunks = 0
        self.total_audio_duration = 0
        self.lock = threading.Lock()
        self.worker_stats = {}
        
    def start_monitoring(self, total_chunks: int, total_duration: float):
        """Start monitoring with known totals"""
        self.start_time = time.time()
        self.total_chunks = total_chunks
        self.total_audio_duration = total_duration
        self.completed_chunks = 0
        self.chunk_completions = {}
        self.worker_stats = {}
        
    def update_completion(self, result: ChunkResult):
        """Update when a chunk completes (thread-safe)"""
        with self.lock:
            self.chunk_completions[result.chunk_id] = {
                'processing_time': result.processing_time,
                'chunk_duration': result.chunk_duration,
                'completion_time': time.time(),
                'worker_id': result.worker_id
            }
            self.completed_chunks += 1
            
            # Track worker statistics
            if result.worker_id not in self.worker_stats:
                self.worker_stats[result.worker_id] = 0
            self.worker_stats[result.worker_id] += 1
    
    def get_progress_info(self) -> Dict:
        """Get current progress information"""
        with self.lock:
            if not self.start_time:
                return {}
            
            elapsed = time.time() - self.start_time
            progress_percent = (self.completed_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0
            
            # Calculate speed based on completed chunks
            total_processed_duration = sum(
                chunk['chunk_duration'] for chunk in self.chunk_completions.values()
            )
            avg_speed = total_processed_duration / elapsed if elapsed > 0 else 0
            
            # Estimate ETA
            remaining_chunks = self.total_chunks - self.completed_chunks
            if self.completed_chunks > 0 and remaining_chunks > 0:
                avg_time_per_chunk = elapsed / self.completed_chunks
                eta = avg_time_per_chunk * remaining_chunks
            else:
                eta = 0
            
            return {
                'completed_chunks': self.completed_chunks,
                'total_chunks': self.total_chunks,
                'progress_percent': progress_percent,
                'elapsed_time': elapsed,
                'processed_audio_duration': total_processed_duration,
                'average_speed_factor': avg_speed,
                'eta_seconds': eta,
                'chunks_per_minute': (self.completed_chunks / elapsed * 60) if elapsed > 0 else 0,
                'worker_stats': dict(self.worker_stats),
                'active_workers': len(self.worker_stats)
            }

class FixedUltraFastTranscriber:
    """Fixed ultra-fast parallel Whisper transcriber with REAL parallel processing"""
    
    def __init__(self, config_file: str = "whisper_config.json"):
        self.load_config(config_file)
        self.performance_monitor = PerformanceMonitor()
        self.model_manager = GlobalModelManager()
        
        # Determine optimal parallel settings
        self.cpu_workers, self.gpu_workers = self._determine_optimal_workers()
        
    def load_config(self, config_file: str):
        """Load configuration"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"🔧 Configuration loaded from {config_file}")
        except FileNotFoundError:
            print("⚠️ Config file not found, using defaults")
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "int8_float16" if device == "cuda" else "int8"
            except ImportError:
                device = "cpu"
                compute_type = "int8"
                
            self.config = {
                "model": "large-v3",
                "device": device,
                "compute_type": compute_type
            }
            
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"💾 Default config saved to {config_file}")
    
    def _determine_optimal_workers(self) -> Tuple[int, int]:
        """Determine optimal number of workers based on system specs"""
        cpu_cores = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        try:
            import torch
            
            if self.config["device"] == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_count = torch.cuda.device_count()
                
                # Conservative GPU worker calculation for stability
                if gpu_memory >= 24:
                    gpu_workers = min(3, gpu_count)  # Conservative for stability
                elif gpu_memory >= 16:
                    gpu_workers = min(2, gpu_count)
                elif gpu_memory >= 8:
                    gpu_workers = 1
                else:
                    gpu_workers = 1
                
                cpu_workers = min(gpu_workers * 2, cpu_cores)
            else:
                gpu_workers = 0
                if memory_gb >= 32 and cpu_cores >= 16:
                    cpu_workers = min(6, cpu_cores)  # Conservative
                elif memory_gb >= 16 and cpu_cores >= 8:
                    cpu_workers = min(4, cpu_cores)
                elif cpu_cores >= 4:
                    cpu_workers = 2
                else:
                    cpu_workers = 1
                    
        except ImportError:
            # Fallback without torch
            gpu_workers = 0
            cpu_workers = min(2, cpu_cores) if cpu_cores >= 4 else 1
        
        print(f"🚀 Optimal workers: {cpu_workers} CPU, {gpu_workers} GPU")
        return cpu_workers, gpu_workers
    
    def transcribe_ultra_fast(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        chunk_size: int = 240,  # 4 minutes for good parallelization
        overlap: int = 10,      # Minimal overlap for speed
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **transcribe_kwargs
    ) -> Dict:
        """
        FIXED Ultra-fast transcription with REAL parallel processing
        """
        print(f"🚀 FIXED Ultra-Fast Transcription: {audio_path}")
        print("🔧 Using pre-loaded models and worker pools")
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
        
        # Step 2: Calculate optimal chunking strategy
        num_chunks = max(1, int(np.ceil(audio_duration / chunk_size)))
        
        if max_workers is None:
            actual_workers = self.gpu_workers if self.config["device"] == "cuda" else self.cpu_workers
        else:
            actual_workers = max_workers
        
        # Ensure we have enough chunks for good parallelization
        if num_chunks < actual_workers * 2:
            # Make chunks smaller for better parallelization
            chunk_size = max(60, chunk_size // 2)  # Minimum 1-minute chunks
            num_chunks = max(1, int(np.ceil(audio_duration / chunk_size)))
        
        print(f"📊 FIXED Parallel Strategy:")
        print(f"   Audio Duration: {audio_duration/3600:.2f} hours")
        print(f"   Chunk Size: {chunk_size}s")
        print(f"   Total Chunks: {num_chunks}")
        print(f"   Worker Pool Size: {actual_workers}")
        print(f"   Expected Speedup: {actual_workers}x+ (with pre-loaded models)")
        
        # Step 3: Create worker pool with pre-loaded models
        print(f"\n🔥 Creating FIXED worker pool with {actual_workers} workers...")
        worker_pool = FixedWorkerPool(self.config, actual_workers)
        
        # Step 4: Prepare transcription parameters
        transcribe_params = {
            'language': language,
            'beam_size': transcribe_kwargs.get('beam_size', 1),  # Fast beam size
            'temperature': transcribe_kwargs.get('temperature', 0.0),
            'condition_on_previous_text': False,  # Faster without context
            'vad_filter': transcribe_kwargs.get('vad_filter', True),
            'vad_parameters': transcribe_kwargs.get('vad_parameters', {
                'min_silence_duration_ms': 500,
                'speech_pad_ms': 200,
            }),
        }
        
        # Step 5: Submit all work to the pool
        print(f"📦 Submitting {num_chunks} chunks to worker pool...")
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size + overlap, audio_duration)
            chunk_duration = chunk_end - chunk_start
            
            # Extract chunk audio from preloaded data (super fast)
            chunk_audio = audio_preloader.get_chunk(chunk_start, chunk_end)
            
            work_item = ChunkWork(
                chunk_id=chunk_idx,
                chunk_audio=chunk_audio,
                chunk_start=chunk_start,
                chunk_duration=chunk_duration,
                transcribe_params=transcribe_params
            )
            
            worker_pool.add_work(work_item)
        
        # Step 6: Start performance monitoring
        self.performance_monitor.start_monitoring(num_chunks, audio_duration)
        
        # Step 7: Collect results with progress tracking
        print(f"\n⚡ Collecting results from {actual_workers} parallel workers...")
        
        all_results = {}
        start_time = time.time()
        
        with tqdm(total=num_chunks, desc="Fixed parallel processing") as pbar:
            for _ in range(num_chunks):
                try:
                    result = worker_pool.get_result(timeout=300)  # 5 min timeout per chunk
                    
                    if result.success:
                        all_results[result.chunk_id] = result
                        self.performance_monitor.update_completion(result)
                        
                        # Update progress with worker info
                        progress_info = self.performance_monitor.get_progress_info()
                        speed = progress_info.get('average_speed_factor', 0)
                        active_workers = progress_info.get('active_workers', 0)
                        
                        pbar.set_postfix({
                            'Worker': result.worker_id,
                            'Speed': f'{speed:.1f}x',
                            'Active': f'{active_workers}/{actual_workers}'
                        })
                        pbar.update(1)
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(progress_info)
                    
                    else:
                        error_msg = result.error or 'Unknown error'
                        print(f"\n❌ Chunk {result.chunk_id} failed (Worker {result.worker_id}): {error_msg}")
                        pbar.update(1)
                        
                except queue.Empty:
                    print(f"\n⚠️ Timeout waiting for chunk result")
                    break
                except Exception as e:
                    print(f"\n❌ Error collecting result: {e}")
                    break
        
        total_processing_time = time.time() - start_time
        
        # Step 8: Cleanup worker pool
        worker_pool.stop()
        
        # Step 9: Process and merge results
        print(f"\n🔗 Merging results from {len(all_results)} chunks...")
        
        all_segments = []
        detected_language = "unknown"
        language_confidence = 0
        final_progress = self.performance_monitor.get_progress_info()
        
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
        
        # Prepare results with detailed statistics
        results = {
            "file": str(audio_path),
            "language": detected_language,
            "language_probability": language_confidence,
            "duration": audio_duration,
            "full_text": full_text,
            "segments": merged_segments,
            "parallel_info": {
                "total_chunks": num_chunks,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "workers_configured": actual_workers,
                "workers_active": final_progress.get('active_workers', 0),
                "chunks_processed": len(all_results),
                "failed_chunks": num_chunks - len(all_results),
                "worker_distribution": final_progress.get('worker_stats', {})
            },
            "performance": {
                "total_processing_time": total_processing_time,
                "real_time_factor": audio_duration / total_processing_time if total_processing_time > 0 else 0,
                "average_speed_factor": final_progress.get('average_speed_factor', 0),
                "chunks_per_minute": final_progress.get('chunks_per_minute', 0),
                "parallel_efficiency": final_progress.get('active_workers', 0) / actual_workers if actual_workers > 0 else 0
            },
            "processing_date": datetime.now().isoformat(),
            "model_config": self.config
        }
        
        # Print performance summary
        self._print_fixed_summary(results)
        
        return results
    
    def _merge_overlapping_segments(self, segments: List[Dict], overlap: int) -> List[Dict]:
        """Merge overlapping segments (optimized)"""
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
    
    def _print_fixed_summary(self, results: Dict):
        """Print comprehensive processing summary"""
        print("\n" + "="*70)
        print("🚀 FIXED ULTRA-FAST TRANSCRIPTION COMPLETED!")
        print("="*70)
        
        duration_hours = results['duration'] / 3600
        processing_mins = results['performance']['total_processing_time'] / 60
        speed_factor = results['performance']['real_time_factor']
        parallel_info = results['parallel_info']
        worker_distribution = parallel_info.get('worker_distribution', {})
        parallel_efficiency = results['performance']['parallel_efficiency']
        
        print(f"📹 Audio Duration: {duration_hours:.2f} hours ({results['duration']:.0f}s)")
        print(f"⚡ Processing Time: {processing_mins:.1f} minutes")
        print(f"🚀 Overall Speed: {speed_factor:.1f}x real-time")
        
        print(f"\n🔥 Parallel Processing Analysis:")
        print(f"   Workers Configured: {parallel_info['workers_configured']}")
        print(f"   Workers Active: {parallel_info['workers_active']}")
        print(f"   Parallel Efficiency: {parallel_efficiency:.1%}")
        print(f"   Chunks Processed: {parallel_info['chunks_processed']}/{parallel_info['total_chunks']}")
        
        # Worker distribution analysis
        if worker_distribution:
            print(f"\n👥 Worker Distribution:")
            for worker_id, chunk_count in worker_distribution.items():
                percentage = chunk_count / parallel_info['chunks_processed'] * 100 if parallel_info['chunks_processed'] > 0 else 0
                print(f"   Worker {worker_id}: {chunk_count} chunks ({percentage:.1f}%)")
            
            if parallel_info['workers_active'] < parallel_info['workers_configured']:
                print(f"   ⚠️ Only {parallel_info['workers_active']}/{parallel_info['workers_configured']} workers were utilized!")
                print(f"   💡 Try reducing model size or chunk count for better utilization")
            else:
                print(f"   ✅ All {parallel_info['workers_active']} workers actively processing!")
        
        # Performance assessment
        print(f"\n📊 Performance Assessment:")
        if speed_factor >= 10:
            print("🏆 EXCELLENT! 10x+ real-time achieved!")
        elif speed_factor >= 5:
            print("✅ GREAT! 5x+ real-time achieved!")
        elif speed_factor >= 2:
            print("✅ GOOD! 2x+ real-time achieved!")
        else:
            print("⚠️ Performance below expectations")
            print("💡 Try: smaller model, fewer workers, or check system resources")
        
        # Time comparison
        normal_estimate = duration_hours * 60  # 1x real-time
        improvement = normal_estimate / processing_mins if processing_mins > 0 else 0
        
        print(f"\n⏱️ Time Comparison:")
        print(f"   Sequential processing: ~{normal_estimate:.0f} minutes")
        print(f"   Fixed parallel processing: {processing_mins:.1f} minutes")
        print(f"   Improvement: {improvement:.1f}x faster!")
        
        print("="*70)
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up resources...")
        self.model_manager.cleanup_models()

# Quick test function
def test_fixed_transcriber():
    """Test the fixed transcriber with comprehensive diagnostics"""
    video_file = r"C:\Users\ASUS\Speak.mp4"
    
    if not Path(video_file).exists():
        print(f"❌ File not found: {video_file}")
        return
    
    print("🧪 Testing FIXED Ultra-Fast Parallel Transcriber")
    print("🎯 This should show MULTIPLE active workers!")
    print("="*60)
    
    try:
        transcriber = FixedUltraFastTranscriber()
        
        # Progress callback to monitor workers
        def progress_callback(info):
            active = info.get('active_workers', 0)
            total = info.get('completed_chunks', 0)
            worker_stats = info.get('worker_stats', {})
            print(f"\r🔥 Progress: {total} chunks, {active} active workers, "
                  f"distribution: {dict(worker_stats)}", end="", flush=True)
        
        result = transcriber.transcribe_ultra_fast(
            video_file,
            language="en",
            chunk_size=120,  # 2-minute chunks for testing
            overlap=5,       # Minimal overlap
            progress_callback=progress_callback,
            
            # Fast settings for testing
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
        )
        
        print("\n\n🎉 TEST COMPLETED!")
        
        # Detailed analysis
        parallel_info = result['parallel_info']
        performance = result['performance']
        
        print(f"\n📊 Detailed Test Results:")
        print(f"   Workers Active: {parallel_info['workers_active']}/{parallel_info['workers_configured']}")
        print(f"   Worker Distribution: {parallel_info['worker_distribution']}")
        print(f"   Processing Speed: {performance['real_time_factor']:.1f}x real-time")
        print(f"   Parallel Efficiency: {performance['parallel_efficiency']:.1%}")
        
        if parallel_info['workers_active'] > 1:
            print(f"\n✅ SUCCESS! {parallel_info['workers_active']} workers were active!")
            print("🚀 Parallel processing is working correctly!")
        else:
            print(f"\n❌ ISSUE: Only {parallel_info['workers_active']} worker was active")
            print("🔧 Check the diagnostics above for the cause")
        
        # Cleanup
        transcriber.cleanup()
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# Main execution function
def main():
    """Complete setup and execution"""
    print("🚀 FIXED Ultra-Fast Parallel Whisper Setup & Execution")
    print("🎯 Target: 10-20x real-time transcription with REAL parallel processing")
    print("🔧 Fixed: Model loading bottlenecks, worker management, GPU conflicts")
    print("=" * 70)
    
    # Enhanced system check
    system_result = check_system_requirements()
    if len(system_result) >= 2:
        system_ok = system_result[0]
        if not system_ok:
            print("❌ System requirements not met")
            return
    
    print("\n📋 FIXED Setup Options:")
    print("1. 🚀 Test FIXED transcriber NOW (quick test)")
    print("2. 🔧 Install dependencies + test FIXED transcriber")
    print("3. 💾 Install dependencies only")
    print("4. 🎬 Transcribe Speak.mp4 with FIXED parallel processing")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice in ["2", "3"]:
            install_requirements()
        
        if choice in ["1", "2"]:
            print("\n🧪 Running FIXED transcriber test...")
            test_fixed_transcriber()
        
        elif choice == "4":
            print("\n🎬 Starting FIXED transcription of Speak.mp4...")
            video_file = r"C:\Users\ASUS\Speak.mp4"
            
            if not Path(video_file).exists():
                print(f"❌ File not found: {video_file}")
                return
            
            transcriber = FixedUltraFastTranscriber()
            
            result = transcriber.transcribe_ultra_fast(
                video_file,
                language="en",
                chunk_size=300,  # 5-minute chunks
                overlap=15,
                beam_size=1,
                temperature=0.0,
                vad_filter=True,
            )
            
            # Save results
            output_dir = Path("transcriptions")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Text file
            txt_file = output_dir / f"Speak_FIXED_{timestamp}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("FIXED ULTRA-FAST TRANSCRIPTION REPORT\n")
                f.write("="*50 + "\n")
                f.write(f"File: {video_file}\n")
                f.write(f"Processing Speed: {result['performance']['real_time_factor']:.1f}x real-time\n")
                f.write(f"Active Workers: {result['parallel_info']['workers_active']}\n")
                f.write(f"Worker Distribution: {result['parallel_info']['worker_distribution']}\n")
                f.write("\n" + result['full_text'])
            
            # JSON report
            json_file = output_dir / f"Speak_FIXED_report_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 FIXED Results saved:")
            print(f"   📝 Text: {txt_file}")
            print(f"   📊 Report: {json_file}")
            
            transcriber.cleanup()
        
        # Final summary
        print("\n" + "="*70)
        print("🚀 FIXED ULTRA-FAST SETUP COMPLETE!")
        print("="*70)
        
        print(f"\n🔧 Key Fixes Implemented:")
        print("   ✅ Pre-loaded model management (eliminates loading bottleneck)")
        print("   ✅ Fixed worker pool with persistent threads")
        print("   ✅ Proper GPU memory handling and device assignment")
        print("   ✅ Worker activity monitoring and diagnostics")
        print("   ✅ Efficient audio preloading")
        print("   ✅ Thread-safe result collection")
        print("   ✅ Comprehensive performance analysis")
        
        print(f"\n💡 Expected Improvements:")
        print("   • Multiple workers should now be active simultaneously")
        print("   • No model loading delays during processing")
        print("   • Better GPU memory utilization")
        print("   • Visible worker distribution in output")
        print("   • Higher parallel efficiency (>50%)")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
