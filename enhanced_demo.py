#!/usr/bin/env python3
"""
Enhanced Whisper demo with performance monitoring and file management
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Import our modules
from transcribe import WhisperTranscriber
from performance_monitor import PerformanceMonitor
from file_manager import TranscriptionFileManager

def enhanced_demo_transcription(audio_file=None, session_name=None, monitor_performance=True):
    """Enhanced demo with performance monitoring and file organization"""
    
    print("🎤 WHISPER LARGE-V3 ENHANCED DEMO")
    print("=" * 50)
    
    # Initialize components
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="auto",
        compute_type="int8",
        languages=["ar", "en"]
    )
    
    file_manager = TranscriptionFileManager()
    
    if monitor_performance:
        performance_monitor = PerformanceMonitor()
    
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
            print("   python enhanced_demo.py <audio_file>")
            print("   python enhanced_demo.py  # (will search for audio files)")
            return
        
        # Select first audio file
        audio_path = audio_files[0]
        print(f"📁 Found {len(audio_files)} audio file(s), using: {audio_path.name}")
    
    # Create session folder
    session_dir = file_manager.create_session_folder(session_name)
    print(f"📂 Session folder: {session_dir}")
    
    # Display file information
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"🎵 Audio file: {audio_path.name}")
    print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Estimate processing time (rough estimate)
    # Large-v3 typically processes at 0.5-1.5x real-time speed
    estimated_duration = file_size_mb * 0.1  # Very rough estimate
    print(f"⏱️  Estimated processing time: {estimated_duration:.1f} seconds")
    
    try:
        # Start performance monitoring
        if monitor_performance:
            performance_monitor.start_monitoring(interval=2)
            print("📈 Performance monitoring started")
        
        print(f"\n🚀 Starting transcription...")
        print("📊 Real-time performance (if monitoring enabled):")
        
        # Transcribe the file
        start_time = datetime.now()
        result = transcriber.transcribe_file(str(audio_path))
        end_time = datetime.now()
        
        # Stop performance monitoring
        performance_data = None
        if monitor_performance:
            performance_summary = performance_monitor.stop_monitoring()
            performance_monitor.print_summary()
            
            # Save detailed performance logs
            performance_monitor.save_detailed_log()
            performance_data = performance_monitor.save_summary_report(
                audio_file=audio_path,
                transcription_result=result
            )
        
        # Organize all files
        print(f"\n📁 Organizing transcription files...")
        organized_files = file_manager.organize_transcription_files(
            audio_file=audio_path,
            transcription_result=result,
            performance_data=performance_summary if monitor_performance else None,
            session_dir=session_dir
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
        print("=" * 50)
        
        # File information
        print(f"🎵 Audio: {audio_path.name}")
        print(f"📊 Duration: {result['duration']:.1f} seconds ({result['duration']/60:.1f} minutes)")
        print(f"🌍 Language: {result['language']} ({result['language_probability']:.1%})")
        print(f"⏱️  Processing time: {result['processing_time']:.1f} seconds")
        print(f"⚡ Speed ratio: {result['speed_ratio']:.1f}x real-time")
        
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
        
        # Transcription preview
        print(f"\n📝 TRANSCRIPTION PREVIEW:")
        segments = result.get('segments', [])
        preview_count = min(5, len(segments))
        
        for i, segment in enumerate(segments[:preview_count]):
            timestamp = f"[{segment['start']:.1f}s-{segment['end']:.1f}s]"
            print(f"   {timestamp} {segment['text']}")
        
        if len(segments) > preview_count:
            print(f"   ... and {len(segments) - preview_count} more segments")
        
        # Recommendations
        if monitor_performance and performance_summary:
            recommendations = performance_monitor._generate_recommendations(performance_summary)
            print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Session summary
        print(f"\n📂 SESSION SUMMARY:")
        print(f"   📁 Session folder: {session_dir}")
        print(f"   📊 Files organized: {len(organized_files)} files")
        print(f"   💾 Total session size: {sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file()) / (1024*1024):.1f} MB")
        
        # Create batch summary (even for single file)
        batch_results = [{
            'audio_file': str(audio_path),
            'success': True,
            'duration': result['duration'],
            'processing_time': result['processing_time'],
            'speed_ratio': result['speed_ratio'],
            'language': result['language'],
            'language_probability': result['language_probability'],
            'organized_files': {k: str(v) for k, v in organized_files.items()}
        }]
        
        summary_file = file_manager.create_batch_summary(session_dir, batch_results)
        print(f"   📋 Batch summary: {summary_file.name}")
        
        return {
            'success': True,
            'session_dir': session_dir,
            'organized_files': organized_files,
            'performance_summary': performance_summary if monitor_performance else None,
            'transcription_result': result
        }
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        
        # Stop monitoring if it was started
        if monitor_performance and 'performance_monitor' in locals():
            performance_monitor.stop_monitoring()
        
        return {
            'success': False,
            'error': str(e)
        }

def batch_transcription(audio_files, session_name=None, monitor_performance=True):
    """Process multiple audio files in batch"""
    
    print("🎤 WHISPER BATCH TRANSCRIPTION")
    print("=" * 50)
    
    # Initialize components
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="auto",
        compute_type="int8",
        languages=["ar", "en"]
    )
    
    file_manager = TranscriptionFileManager()
    
    if monitor_performance:
        performance_monitor = PerformanceMonitor()
    
    # Create session folder
    session_dir = file_manager.create_session_folder(session_name)
    print(f"📂 Session folder: {session_dir}")
    
    batch_results = []
    total_files = len(audio_files)
    
    try:
        # Start performance monitoring for entire batch
        if monitor_performance:
            performance_monitor.start_monitoring(interval=2)
            print("📈 Performance monitoring started for batch")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n🎵 Processing file {i}/{total_files}: {audio_file}")
            
            try:
                # Transcribe the file
                result = transcriber.transcribe_file(str(audio_file))
                
                # Organize files
                organized_files = file_manager.organize_transcription_files(
                    audio_file=Path(audio_file),
                    transcription_result=result,
                    session_dir=session_dir
                )
                
                batch_results.append({
                    'audio_file': str(audio_file),
                    'success': True,
                    'duration': result['duration'],
                    'processing_time': result['processing_time'],
                    'speed_ratio': result['speed_ratio'],
                    'language': result['language'],
                    'language_probability': result['language_probability'],
                    'organized_files': {k: str(v) for k, v in organized_files.items()}
                })
                
                print(f"✅ Completed: {result['duration']:.1f}s audio, {result['processing_time']:.1f}s processing")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                batch_results.append({
                    'audio_file': str(audio_file),
                    'success': False,
                    'error': str(e)
                })
        
        # Stop performance monitoring
        if monitor_performance:
            performance_summary = performance_monitor.stop_monitoring()
            performance_monitor.print_summary()
        
        # Create batch summary
        summary_file = file_manager.create_batch_summary(session_dir, batch_results)
        
        # Print batch results
        successful = len([r for r in batch_results if r['success']])
        failed = total_files - successful
        
        print(f"\n📊 BATCH RESULTS:")
        print(f"   ✅ Successful: {successful}/{total_files}")
        print(f"   ❌ Failed: {failed}/{total_files}")
        print(f"   📋 Summary: {summary_file}")
        
        return {
            'success': True,
            'session_dir': session_dir,
            'batch_results': batch_results,
            'performance_summary': performance_summary if monitor_performance else None
        }
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced Whisper transcription with performance monitoring")
    parser.add_argument('audio_files', nargs='*', help='Audio files to transcribe')
    parser.add_argument('--session', '-s', help='Session name for organizing files')
    parser.add_argument('--no-monitor', action='store_true', help='Disable performance monitoring')
    parser.add_argument('--batch', '-b', action='store_true', help='Process multiple files in batch mode')
    
    args = parser.parse_args()
    
    # Check if we should monitor performance
    monitor_performance = not args.no_monitor
    
    if args.batch or len(args.audio_files) > 1:
        # Batch processing
        if not args.audio_files:
            print("❌ No audio files specified for batch processing")
            return
        
        batch_transcription(args.audio_files, args.session, monitor_performance)
    else:
        # Single file processing
        audio_file = args.audio_files[0] if args.audio_files else None
        enhanced_demo_transcription(audio_file, args.session, monitor_performance)

if __name__ == "__main__":
    main()
