#!/usr/bin/env python3
"""
Simple example of using the Whisper transcription setup
"""

import os
from pathlib import Path
from transcribe import WhisperTranscriber

def demo_transcription():
    """Demonstrate transcription with sample usage"""
    
    print("🎤 WHISPER LARGE-V3 DEMO")
    print("=" * 40)
    
    # Initialize transcriber with your optimal settings
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="auto",           # Auto-detect best device
        compute_type="int8",     # Your preference for speed
        languages=["ar", "en"]   # Arabic and English
    )
    
    # Look for audio files in current directory
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(".").glob(f"*{ext}"))
    
    if not audio_files:
        print("📁 No audio files found in current directory")
        print("\n💡 To test transcription:")
        print("1. Place an audio file (.mp3, .wav, .m4a, etc.) in this directory")
        print("2. Run: python demo.py")
        print("3. Or use: python transcribe.py <audio_file>")
        print("\n📋 Example commands:")
        print("   python transcribe.py my_audio.mp3")
        print("   python transcribe.py my_audio.wav json")
        print("   python transcribe.py my_audio.m4a srt")
        return
    
    print(f"📁 Found {len(audio_files)} audio file(s):")
    for i, file in enumerate(audio_files, 1):
        print(f"   {i}. {file.name}")
    
    # Ask user which file to transcribe
    if len(audio_files) == 1:
        selected_file = audio_files[0]
        print(f"\n🎵 Transcribing: {selected_file.name}")
    else:
        print(f"\n🎵 Transcribing first file: {audio_files[0].name}")
        print("   (You can specify others manually)")
        selected_file = audio_files[0]
    
    try:
        # Transcribe the file
        result = transcriber.transcribe_file(str(selected_file))
        
        # Save results in multiple formats
        base_name = selected_file.stem
        
        # Save JSON (detailed)
        json_file = f"{base_name}_transcription.json"
        import json
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save TXT (plain text)
        txt_file = f"{base_name}_transcription.txt"
        transcriber.save_text_output(result, txt_file)
        
        # Save SRT (subtitles)
        srt_file = f"{base_name}_transcription.srt"
        transcriber.save_srt_output(result, srt_file)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 {json_file} (detailed JSON)")
        print(f"   📝 {txt_file} (plain text)")
        print(f"   🎬 {srt_file} (subtitles)")
        
        # Show summary
        print(f"\n📊 TRANSCRIPTION SUMMARY:")
        print(f"   🎵 File: {selected_file.name}")
        print(f"   🌍 Language: {result['language']} ({result['language_probability']:.1%})")
        print(f"   ⏱️  Duration: {result['duration']:.1f} seconds")
        print(f"   🚀 Processing: {result['processing_time']:.1f} seconds")
        print(f"   ⚡ Speed: {result['speed_ratio']:.1f}x real-time")
        
        # Show first few lines of transcription
        print(f"\n📝 PREVIEW:")
        for i, segment in enumerate(result['segments'][:3]):
            print(f"   [{segment['start']:.1f}s] {segment['text']}")
        
        if len(result['segments']) > 3:
            print(f"   ... and {len(result['segments']) - 3} more segments")
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")

if __name__ == "__main__":
    demo_transcription()
