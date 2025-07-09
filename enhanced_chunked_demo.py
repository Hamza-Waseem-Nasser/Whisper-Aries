#!/usr/bin/env python3
"""
Enhanced Whisper demo with chunked/parallel processing for large files
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

# Import our modules
from transcribe import WhisperTranscriber
from optimized_chunked_transcriber import OptimizedChunkedTranscriber
from simple_parallel_transcriber import SimpleParallelTranscriber
from performance_monitor import PerformanceMonitor
from file_manager import TranscriptionFileManager

class ProgressTracker:
    """Track and display transcription progress"""
    
    def __init__(self, total_chunks=None):
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self.start_time = time.time()
        self.last_update = time.time()
    
    def update(self, completed, message):
        """Update progress"""
        self.completed_chunks = completed
        current_time = time.time()
        
        # Only print every 5 seconds or on completion
        if current_time - self.last_update >= 5 or completed == self.total_chunks:
            elapsed = current_time - self.start_time
            
            if self.total_chunks:
                progress_percent = (completed / self.total_chunks) * 100
                print(f"📊 Progress: {completed}/{self.total_chunks} ({progress_percent:.1f}%) - {message}")
                
                if completed > 0:
                    estimated_total = elapsed * (self.total_chunks / completed)
                    remaining = estimated_total - elapsed
                    print(f"⏱️  Estimated remaining: {remaining/60:.1f} minutes")
            else:
                print(f"📊 {message}")
            
            self.last_update = current_time

def should_use_chunked_processing(audio_path, duration_threshold_minutes=10):
    """
    Determine if we should use chunked processing based on file size/duration
    
    Args:
        audio_path: Path to audio file
        duration_threshold_minutes: Threshold for using chunked processing
    
    Returns:
        bool: True if chunked processing should be used
    """
    try:
        # Check file size (rough heuristic)
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        # Rough estimate: 1 MB ≈ 1 minute of audio (for compressed formats)
        estimated_duration = file_size_mb
        
        return estimated_duration > duration_threshold_minutes
    except:
        return False

def enhanced_transcription_with_chunking(audio_file=None, session_name=None, 
                                       monitor_performance=True, force_chunked=False,
                                       chunk_length_minutes=5, max_workers=None):
    """Enhanced transcription with automatic chunked processing for large files"""
    
    print("🎤 WHISPER LARGE-V3 ENHANCED DEMO")
    print("🚀 WITH CHUNKED/PARALLEL PROCESSING")
    print("=" * 60)
    
    # Find audio file
    if audio_file:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_file}")
            return
    else:
        # Look for audio files in current directory
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(".").glob(f"*{ext}"))
        
        if not audio_files:
            print("📁 No audio files found in current directory")
            print("\n💡 Usage:")
            print("   python enhanced_chunked_demo.py <audio_file>")
            print("   python enhanced_chunked_demo.py  # (will search for audio files)")
            return
        
        # Select first audio file
        audio_path = audio_files[0]
        print(f"📁 Found {len(audio_files)} audio file(s), using: {audio_path.name}")
    
    # Initialize file manager
    file_manager = TranscriptionFileManager()
    session_dir = file_manager.create_session_folder(session_name)
    print(f"📂 Session folder: {session_dir}")
    
    # Display file information
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"🎵 Audio file: {audio_path.name}")
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Determine processing method
    use_chunked = force_chunked or should_use_chunked_processing(audio_path)
    
    print(f"\n🤖 Processing method: {'CHUNKED/PARALLEL' if use_chunked else 'STANDARD'}")
    
    if use_chunked:
        print(f"📏 Chunk length: {chunk_length_minutes} minutes")
        print(f"👥 Max workers: {max_workers or 'auto'}")
        print("⚡ Expected significant speed improvement!")
    
    # Initialize performance monitoring
    performance_monitor = None
    if monitor_performance:
        performance_monitor = PerformanceMonitor()
    
    try:
        # Start performance monitoring
        if monitor_performance:
            performance_monitor.start_monitoring(interval=2)
            print("📈 Performance monitoring started")
        
        print(f"\n🚀 Starting transcription...")
        start_time = datetime.now()
        
        if use_chunked:
            # Try optimized chunked processing first, fallback to simple parallel
            try:
                transcriber = OptimizedChunkedTranscriber(
                    model_size="large-v3",
                    device="auto",
                    compute_type="int8",
                    chunk_length_minutes=chunk_length_minutes,
                    overlap_seconds=5,
                    max_workers=max_workers
                )
                
                # Set up progress tracking
                progress_tracker = ProgressTracker()
                
                # Transcribe with chunked processing
                result = transcriber.transcribe_file(audio_path)
                
                # Check if chunking actually worked (non-zero processing time)
                if result.get('total_processing_time', 0) < 1:
                    raise Exception("Chunking failed, falling back to simple parallel")
                    
            except Exception as e:
                print(f"⚠️ Optimized chunking failed: {e}")
                print("🔄 Falling back to simple parallel processing...")
                
                transcriber = SimpleParallelTranscriber(
                    model_size="large-v3",
                    device="auto",
                    compute_type="int8",
                    chunk_length_minutes=chunk_length_minutes,
                    max_workers=max_workers
                )
                
                result = transcriber.transcribe_file(audio_path)
            
        else:
            # Use standard processing
            transcriber = WhisperTranscriber(
                model_size="large-v3",
                device="auto",
                compute_type="int8",
                languages=["ar", "en"]
            )
            
            result = transcriber.transcribe_file(str(audio_path))
        
        end_time = datetime.now()
        
        # Stop performance monitoring
        performance_summary = None
        if monitor_performance:
            performance_summary = performance_monitor.stop_monitoring()
            performance_monitor.print_summary()
            
            # Save detailed performance logs
            performance_monitor.save_detailed_log()
            performance_data = performance_monitor.save_summary_report(
                audio_file=audio_path,
                transcription_result=result
            )
        
        # Organize all files (don't copy large audio files)
        print(f"\n📁 Organizing transcription files...")
        organized_files = file_manager.organize_transcription_files(
            audio_file=audio_path,
            transcription_result=result,
            performance_data=performance_summary if monitor_performance else None,
            session_dir=session_dir,
            copy_audio=False  # Don't copy large audio files
        )
        
        # Also save a simple text file in the current directory for easy access
        simple_text_file = f"{audio_path.stem}_transcription.txt"
        with open(simple_text_file, 'w', encoding='utf-8') as f:
            # Extract text from segments if 'text' field is not available
            if 'text' in result:
                f.write(result['text'])
            else:
                # Build text from segments
                full_text = ""
                for segment in result.get('segments', []):
                    full_text += segment['text'] + " "
                f.write(full_text.strip())
        
        print(f"📝 Simple text file saved: {simple_text_file}")
        
        # Print results summary
        print(f"\n✅ TRANSCRIPTION COMPLETED!")
        print("=" * 60)
        
        # File information
        print(f"🎵 Audio: {audio_path.name}")
        print(f"📊 Duration: {result['duration']:.1f} seconds ({result['duration']/60:.1f} minutes)")
        print(f"🌍 Language: {result['language']} ({result.get('language_probability', 0):.1%})")
        print(f"⏱️  Processing time: {result['processing_time']:.1f} seconds")
        print(f"⚡ Speed ratio: {result['speed_ratio']:.1f}x real-time")
        
        # Chunked processing stats
        if use_chunked and 'segments_processed' in result:
            print(f"📦 Segments processed: {result['segments_succeeded']}/{result['segments_processed']}")
        
        # Performance summary
        if monitor_performance and performance_summary:
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print(f"🖥️  CPU usage: {performance_summary['cpu_avg']:.1f}% avg, {performance_summary['cpu_max']:.1f}% max")
            print(f"💾 RAM usage: {performance_summary['memory_avg']:.1f}% avg, {performance_summary['memory_max']:.1f}% max")
            print(f"🎮 GPU usage: {performance_summary['gpu_load_avg']:.1f}% avg, {performance_summary['gpu_load_max']:.1f}% max")
            print(f"🎯 GPU memory: {performance_summary['gpu_memory_avg']:.1f}% avg, {performance_summary['gpu_memory_max']:.1f}% max")
            print(f"🔧 GPU: {performance_summary['gpu_name']}")
        
        # Files created
        print(f"\n💾 FILES CREATED:")
        for file_type, file_path in organized_files.items():
            print(f"   📄 {file_type.upper()}: {file_path.name}")
        print(f"   📝 SIMPLE TEXT: {simple_text_file}")
        
        print(f"\n📂 Session folder: {session_dir}")
        
        # Expert recommendations
        print(f"\n🧠 EXPERT RECOMMENDATIONS:")
        
        if use_chunked:
            speed_improvement = result['speed_ratio']
            if speed_improvement > 5:
                print("🎯 Excellent speed! Consider slightly longer chunks for even better efficiency.")
            elif speed_improvement > 2:
                print("✅ Good speed ratio. Current settings are well optimized.")
            else:
                print("⚠️  Consider shorter chunks or more workers for better parallelization.")
                
            print(f"🔧 Current chunk size: {chunk_length_minutes} min (try 3-7 min for optimal balance)")
            print(f"👥 Current workers: {transcriber.max_workers} (try {min(6, os.cpu_count())} for more parallelism)")
        else:
            if result['speed_ratio'] < 1:
                print("🚀 Consider using chunked processing (--force-chunked) for better speed!")
            
        if performance_summary:
            if performance_summary['gpu_load_avg'] < 50:
                print("🎮 GPU is underutilized. Consider larger batch sizes or more workers.")
            if performance_summary['memory_avg'] < 70:
                print("💾 RAM usage is low. Can handle larger chunks or more parallel workers.")
        
        print("🎯 For maximum speed: Use 3-5 minute chunks with 4-6 workers on files >10 minutes")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Transcription interrupted by user")
        if monitor_performance:
            performance_monitor.stop_monitoring()
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        if monitor_performance:
            performance_monitor.stop_monitoring()
        raise

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Enhanced Whisper transcription with chunked processing")
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--session-name", help="Custom session name")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable performance monitoring")
    parser.add_argument("--force-chunked", action="store_true", help="Force chunked processing even for small files")
    parser.add_argument("--chunk-length", type=int, default=5, help="Chunk length in minutes (default: 5)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    enhanced_transcription_with_chunking(
        audio_file=args.audio_file,
        session_name=args.session_name,
        monitor_performance=not args.no_monitoring,
        force_chunked=args.force_chunked,
        chunk_length_minutes=args.chunk_length,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()
