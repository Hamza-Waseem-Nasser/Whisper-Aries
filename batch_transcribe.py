#!/usr/bin/env python3
"""
Batch transcription script for multiple audio files
Optimized for Arabic and English with INT8 quantization
"""

import os
import sys
import json
from pathlib import Path
from transcribe import WhisperTranscriber

def batch_transcribe_directory(input_dir=".", output_dir="transcriptions"):
    """
    Transcribe all audio files in a directory
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save transcriptions
    """
    
    print("🎤 BATCH TRANSCRIPTION - WHISPER LARGE-V3")
    print("=" * 50)
    
    # Find all audio files
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.webm']
    audio_files = []
    
    input_path = Path(input_dir)
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"📁 No audio files found in: {input_dir}")
        print(f"   Looking for: {', '.join(audio_extensions)}")
        return
    
    print(f"📁 Found {len(audio_files)} audio files in: {input_dir}")
    for i, file in enumerate(audio_files, 1):
        size_mb = file.stat().st_size / (1024*1024)
        print(f"   {i:2d}. {file.name:<30} ({size_mb:.1f} MB)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize transcriber
    print(f"\n🔧 Initializing Whisper Large-v3...")
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        languages=["ar", "en"]
    )
    
    # Process each file
    results = []
    total_duration = 0
    total_processing_time = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        print(f"{'='*50}")
        
        try:
            # Transcribe
            result = transcriber.transcribe_file(str(audio_file))
            results.append(result)
            
            # Update statistics
            total_duration += result['duration']
            total_processing_time += result['processing_time']
            
            # Save individual results
            base_name = audio_file.stem
            
            # JSON (detailed)
            json_file = Path(output_dir) / f"{base_name}_transcription.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # TXT (plain text)
            txt_file = Path(output_dir) / f"{base_name}_transcription.txt"
            transcriber.save_text_output(result, txt_file)
            
            # SRT (subtitles)
            srt_file = Path(output_dir) / f"{base_name}_transcription.srt"
            transcriber.save_srt_output(result, srt_file)
            
            print(f"💾 Saved: {json_file.name}, {txt_file.name}, {srt_file.name}")
            
        except Exception as e:
            print(f"❌ Failed to process {audio_file.name}: {e}")
            results.append({
                "file": str(audio_file),
                "error": str(e)
            })
    
    # Create summary report
    print(f"\n{'='*50}")
    print("📊 BATCH TRANSCRIPTION SUMMARY")
    print(f"{'='*50}")
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"✅ Successfully processed: {len(successful)} files")
    print(f"❌ Failed: {len(failed)} files")
    
    if successful:
        print(f"\n📈 PERFORMANCE STATISTICS:")
        print(f"   🎵 Total audio duration: {total_duration/60:.1f} minutes")
        print(f"   ⏱️  Total processing time: {total_processing_time/60:.1f} minutes")
        print(f"   ⚡ Average speed: {total_duration/total_processing_time:.1f}x real-time")
        
        # Language statistics
        languages = {}
        for result in successful:
            lang = result['language']
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\n🌍 LANGUAGE DETECTION:")
        for lang, count in languages.items():
            print(f"   {lang}: {count} files")
    
    if failed:
        print(f"\n❌ FAILED FILES:")
        for result in failed:
            print(f"   {Path(result['file']).name}: {result['error']}")
    
    # Save batch summary
    summary_file = Path(output_dir) / "batch_summary.json"
    summary = {
        "total_files": len(audio_files),
        "successful": len(successful),
        "failed": len(failed),
        "total_duration": total_duration,
        "total_processing_time": total_processing_time,
        "average_speed": total_duration/total_processing_time if total_processing_time > 0 else 0,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Batch summary saved: {summary_file}")
    print(f"📁 All transcriptions saved in: {output_dir}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "."
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "transcriptions"
    
    batch_transcribe_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
