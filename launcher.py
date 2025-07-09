#!/usr/bin/env python3
"""
Simple launcher for Whisper transcription with performance monitoring
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def main():
    """Main launcher function"""
    
    print("🎤 WHISPER TRANSCRIPTION LAUNCHER")
    print("=" * 50)
    
    # Check what's available
    try:
        from transcribe import WhisperTranscriber
        print("✅ Whisper transcriber available")
    except ImportError as e:
        print(f"❌ Whisper transcriber not available: {e}")
        return
    
    # Check for performance monitoring
    performance_available = False
    try:
        from performance_monitor import PerformanceMonitor
        print("✅ Advanced performance monitoring available")
        performance_available = True
    except ImportError:
        try:
            from basic_performance_monitor import BasicPerformanceMonitor
            print("✅ Basic performance monitoring available")
            performance_available = "basic"
        except ImportError:
            print("⚠️  No performance monitoring available")
    
    # Check for file management
    file_management_available = False
    try:
        from file_manager import TranscriptionFileManager
        print("✅ File management available")
        file_management_available = True
    except ImportError:
        print("⚠️  File management not available")
    
    # Get audio file
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
        if not audio_file.exists():
            print(f"❌ Audio file not found: {audio_file}")
            return
    else:
        # Look for audio files
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(".").glob(f"*{ext}"))
        
        if not audio_files:
            print("❌ No audio files found")
            print("Usage: python launcher.py <audio_file>")
            return
        
        audio_file = audio_files[0]
        print(f"📁 Using audio file: {audio_file.name}")
    
    # Initialize components
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="auto",
        compute_type="int8",
        languages=["ar", "en"]
    )
    
    if file_management_available:
        file_manager = TranscriptionFileManager()
        session_dir = file_manager.create_session_folder()
        print(f"📂 Session folder: {session_dir}")
    
    # Start performance monitoring
    if performance_available == True:
        monitor = PerformanceMonitor()
        monitor.start_monitoring(interval=2)
        print("📈 Advanced performance monitoring started")
    elif performance_available == "basic":
        monitor = BasicPerformanceMonitor()
        monitor.start_monitoring(interval=3)
        print("📈 Basic performance monitoring started")
    else:
        monitor = None
        print("⚠️  No performance monitoring")
    
    try:
        # Display file info
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        print(f"\n🎵 Transcribing: {audio_file.name}")
        print(f"📊 File size: {file_size_mb:.1f} MB")
        print(f"🚀 Starting transcription...")
        
        if monitor:
            print("📊 Real-time performance:")
        
        # Transcribe
        result = transcriber.transcribe_file(str(audio_file))
        
        # Stop monitoring
        if monitor:
            performance_summary = monitor.stop_monitoring()
            monitor.print_summary()
        
        # Organize files using file manager (this handles all file creation)
        if file_management_available:
            organized_files = file_manager.organize_transcription_files(
                audio_file=audio_file,
                transcription_result=result,
                performance_data=performance_summary if monitor else None,
                session_dir=session_dir
            )
            
            # Also create a simple text file in the current directory for easy access
            simple_txt_file = f"{audio_file.stem}_transcription.txt"
            with open(simple_txt_file, 'w', encoding='utf-8') as f:
                # Extract text from segments if 'text' field is not available
                if 'text' in result and result['text']:
                    f.write(result['text'])
                else:
                    # Build text from segments
                    full_text = ""
                    for segment in result.get('segments', []):
                        full_text += segment['text'] + " "
                    f.write(full_text.strip())
            
            print(f"\n📁 Files organized in: {session_dir}")
            print(f"📝 Simple text file: {simple_txt_file}")
        else:
            # Fallback: save files in current directory if file manager not available
            base_name = audio_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON
            json_file = f"{base_name}_{timestamp}.json"
            import json
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save TXT (simple transcription)
            txt_file = f"{base_name}_{timestamp}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                # Extract text from segments if 'text' field is not available
                if 'text' in result and result['text']:
                    f.write(result['text'])
                else:
                    # Build text from segments
                    full_text = ""
                    for segment in result.get('segments', []):
                        full_text += segment['text'] + " "
                    f.write(full_text.strip())
            
            # Save simple text file
            simple_txt_file = f"{base_name}_transcription.txt"
            with open(simple_txt_file, 'w', encoding='utf-8') as f:
                if 'text' in result and result['text']:
                    f.write(result['text'])
                else:
                    full_text = ""
                    for segment in result.get('segments', []):
                        full_text += segment['text'] + " "
                    f.write(full_text.strip())
            
            # Save SRT
            srt_file = f"{base_name}_{timestamp}.srt"
            segments = result.get('segments', [])
            with open(srt_file, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start = seconds_to_srt_time(segment['start'])
                    end = seconds_to_srt_time(segment['end'])
                    f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
            
            organized_files = {
                'json': json_file,
                'txt': txt_file,
                'simple_txt': simple_txt_file,
                'srt': srt_file
            }
        
        # Save performance log
        if monitor:
            if hasattr(monitor, 'save_performance_log'):
                monitor.save_performance_log(f"{audio_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_performance.json")
        
        # Print results
        print(f"\n✅ TRANSCRIPTION COMPLETED!")
        print("=" * 50)
        print(f"🎵 File: {audio_file.name}")
        print(f"📊 Duration: {result['duration']:.1f} seconds")
        print(f"🌍 Language: {result['language']} ({result['language_probability']:.1%})")
        print(f"⏱️  Processing: {result['processing_time']:.1f} seconds")
        print(f"⚡ Speed: {result['speed_ratio']:.1f}x real-time")
        
        print(f"\n💾 Files created:")
        if file_management_available:
            print(f"   📁 Session folder: {session_dir.name}")
            print(f"   📝 Simple text: {audio_file.stem}_transcription.txt (current directory)")
            print(f"   📄 Detailed files in session folder:")
            for file_type, file_path in organized_files.items():
                if file_type != 'metadata':  # Skip metadata in display
                    print(f"      - {file_type.upper()}: {Path(file_path).name}")
        else:
            print(f"   📄 {organized_files['json']} (detailed)")
            print(f"   📝 {organized_files['txt']} (timestamped)")
            print(f"   📝 {organized_files['simple_txt']} (simple text)")
            print(f"   🎬 {organized_files['srt']} (subtitles)")
        
        # Preview
        print(f"\n📝 Preview:")
        for i, segment in enumerate(result.get('segments', [])[:3]):
            print(f"   [{segment['start']:.1f}s] {segment['text']}")
        
        if len(result.get('segments', [])) > 3:
            print(f"   ... and {len(result.get('segments', [])) - 3} more segments")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if monitor:
            monitor.stop_monitoring()

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

if __name__ == "__main__":
    main()
