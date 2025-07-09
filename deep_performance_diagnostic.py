#!/usr/bin/env python3
"""
Deep Performance Diagnostic for Whisper GPU Usage
"""

import os
import time
import threading
import torch
import psutil
from pathlib import Path
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class PerformanceDiagnostic:
    """Deep performance analysis for Whisper"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.monitoring = False
        self.stats = []
        
    def check_system_setup(self):
        """Check system configuration"""
        print("🔍 SYSTEM DIAGNOSTIC")
        print("=" * 50)
        
        # PyTorch and CUDA info
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        
        # CPU info
        print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
        # Faster Whisper
        if WHISPER_AVAILABLE:
            print("✅ faster-whisper available")
        else:
            print("❌ faster-whisper not available")
            
        # GPU utilities
        if self.gpu_available:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU {gpu.id}: {gpu.name}")
                print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
                print(f"  Load: {gpu.load * 100:.1f}%")
        
        print()
    
    def test_model_loading(self):
        """Test model loading performance"""
        if not WHISPER_AVAILABLE:
            print("❌ Cannot test model loading - faster-whisper not available")
            return
            
        print("🧪 MODEL LOADING TEST")
        print("=" * 50)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {device}...")
        
        start_time = time.time()
        try:
            model = WhisperModel("large-v3", device=device, compute_type="int8")
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU memory used: {memory_used:.1f} GB")
                
            return model
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return None
    
    def test_single_transcription(self, model, audio_file):
        """Test single file transcription"""
        if model is None or not Path(audio_file).exists():
            print("❌ Cannot test transcription - model or audio file unavailable")
            return
            
        print("🎵 SINGLE TRANSCRIPTION TEST")
        print("=" * 50)
        
        print(f"Transcribing: {Path(audio_file).name}")
        
        # Monitor during transcription
        self.start_monitoring()
        
        start_time = time.time()
        try:
            segments, info = model.transcribe(
                audio_file,
                beam_size=1,  # Reduce beam size for speed test
                temperature=0.0,
                vad_filter=False,  # Disable VAD for speed
                word_timestamps=False  # Disable word timestamps for speed
            )
            
            # Convert segments to list to actually process them
            segments_list = list(segments)
            
            processing_time = time.time() - start_time
            
            self.stop_monitoring()
            
            speed_ratio = info.duration / processing_time if processing_time > 0 else 0
            
            print(f"✅ Transcription completed:")
            print(f"  Audio duration: {info.duration:.1f} seconds")
            print(f"  Processing time: {processing_time:.1f} seconds")
            print(f"  Speed ratio: {speed_ratio:.1f}x real-time")
            print(f"  Language: {info.language} (prob: {info.language_probability:.2f})")
            print(f"  Segments: {len(segments_list)}")
            
            # Show performance stats
            self.show_performance_summary()
            
        except Exception as e:
            self.stop_monitoring()
            print(f"❌ Transcription failed: {e}")
    
    def test_threading_behavior(self, model, audio_file):
        """Test how model behaves in threading environment"""
        if model is None or not Path(audio_file).exists():
            print("❌ Cannot test threading - model or audio file unavailable")
            return
            
        print("🧵 THREADING BEHAVIOR TEST")
        print("=" * 50)
        
        def transcribe_worker(worker_id):
            print(f"Worker {worker_id} starting...")
            start_time = time.time()
            
            try:
                # Check if model is actually on GPU for this thread
                if hasattr(model, 'model') and hasattr(model.model, 'device'):
                    print(f"Worker {worker_id} - Model device: {model.model.device}")
                
                segments, info = model.transcribe(
                    audio_file,
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=False,
                    word_timestamps=False
                )
                
                # Process segments
                segments_list = list(segments)
                processing_time = time.time() - start_time
                speed_ratio = info.duration / processing_time if processing_time > 0 else 0
                
                print(f"Worker {worker_id} completed in {processing_time:.1f}s ({speed_ratio:.1f}x)")
                
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
        
        # Test with 2 threads (like our chunked transcriber)
        print("Testing with 2 concurrent threads...")
        self.start_monitoring()
        
        threads = []
        for i in range(2):
            thread = threading.Thread(target=transcribe_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.stop_monitoring()
        self.show_performance_summary()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.stats = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor system performance"""
        while self.monitoring:
            stat = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
            }
            
            if torch.cuda.is_available():
                stat['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                stat['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
            
            if self.gpu_available:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    stat['gpu_load'] = gpu.load * 100
                    stat['gpu_memory_used'] = gpu.memoryUsed
            
            self.stats.append(stat)
            time.sleep(0.5)  # Monitor every 0.5 seconds
    
    def show_performance_summary(self):
        """Show performance summary"""
        if not self.stats:
            print("No performance data collected")
            return
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 30)
        
        cpu_avg = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)
        memory_avg = sum(s['memory_percent'] for s in self.stats) / len(self.stats)
        
        print(f"Average CPU usage: {cpu_avg:.1f}%")
        print(f"Average Memory usage: {memory_avg:.1f}%")
        
        if torch.cuda.is_available():
            gpu_mem_avg = sum(s.get('gpu_memory_allocated', 0) for s in self.stats) / len(self.stats)
            gpu_mem_max = max(s.get('gpu_memory_allocated', 0) for s in self.stats)
            print(f"Average GPU memory: {gpu_mem_avg:.1f} GB")
            print(f"Peak GPU memory: {gpu_mem_max:.1f} GB")
        
        if self.gpu_available and any('gpu_load' in s for s in self.stats):
            gpu_loads = [s['gpu_load'] for s in self.stats if 'gpu_load' in s]
            if gpu_loads:
                gpu_load_avg = sum(gpu_loads) / len(gpu_loads)
                gpu_load_max = max(gpu_loads)
                print(f"Average GPU load: {gpu_load_avg:.1f}%")
                print(f"Peak GPU load: {gpu_load_max:.1f}%")
    
    def run_full_diagnostic(self, audio_file):
        """Run complete diagnostic"""
        print(f"🚀 WHISPER PERFORMANCE DIAGNOSTIC")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎵 Test file: {Path(audio_file).name}")
        print("=" * 60)
        
        # System check
        self.check_system_setup()
        
        # Model loading
        model = self.test_model_loading()
        
        if model:
            # Single transcription test
            self.test_single_transcription(model, audio_file)
            print()
            
            # Threading test
            self.test_threading_behavior(model, audio_file)
        
        print("\n✅ Diagnostic complete!")


def main():
    import sys
    
    # Find a test audio file
    test_files = [
        "example.mp3",
        "harvard.wav", 
        "examples/example.mp3"
    ]
    
    audio_file = None
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        for test_file in test_files:
            if Path(test_file).exists():
                audio_file = test_file
                break
    
    if not audio_file or not Path(audio_file).exists():
        print("❌ No audio file found for testing")
        print("Usage: python deep_performance_diagnostic.py [audio_file]")
        print("Or place example.mp3 or harvard.wav in the current directory")
        return
    
    diagnostic = PerformanceDiagnostic()
    diagnostic.run_full_diagnostic(audio_file)


if __name__ == "__main__":
    main()
