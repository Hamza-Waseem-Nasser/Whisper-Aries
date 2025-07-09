#!/usr/bin/env python3
"""
Diagnostic script to identify FFmpeg issues
"""

import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

def diagnose_ffmpeg():
    """Comprehensive FFmpeg diagnosis"""
    
    print("🔍 FFMPEG DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # 1. Check Python environment
    print("\n1. PYTHON ENVIRONMENT:")
    print(f"   📍 Python executable: {sys.executable}")
    print(f"   📁 Current working directory: {os.getcwd()}")
    print(f"   🔧 Virtual env: {os.environ.get('VIRTUAL_ENV', 'None')}")
    
    # 2. Check PATH environment
    print("\n2. ENVIRONMENT PATH:")
    path_entries = os.environ.get('PATH', '').split(os.pathsep)
    ffmpeg_paths = [p for p in path_entries if 'ffmpeg' in p.lower()]
    print(f"   📋 Total PATH entries: {len(path_entries)}")
    print(f"   🎬 FFmpeg-related paths: {ffmpeg_paths}")
    
    # 3. Check FFmpeg availability methods
    print("\n3. FFMPEG AVAILABILITY TESTS:")
    
    # Test 1: which/where command
    try:
        result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"   ✅ 'where ffmpeg': {result.stdout.strip()}")
        else:
            print(f"   ❌ 'where ffmpeg' failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ 'where ffmpeg' exception: {e}")
    
    # Test 2: Direct subprocess call
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Direct subprocess 'ffmpeg -version': Success")
            print(f"      📝 First line: {result.stdout.split()[0:4]}")
        else:
            print(f"   ❌ Direct subprocess failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Direct subprocess exception: {e}")
    
    # Test 3: Shell=True subprocess call
    try:
        result = subprocess.run('ffmpeg -version', capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"   ✅ Shell subprocess 'ffmpeg -version': Success")
        else:
            print(f"   ❌ Shell subprocess failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Shell subprocess exception: {e}")
    
    # Test 4: Full path resolution
    try:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            print(f"   ✅ shutil.which('ffmpeg'): {ffmpeg_path}")
            
            # Test with full path
            result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ Full path subprocess: Success")
            else:
                print(f"   ❌ Full path subprocess failed: {result.stderr}")
        else:
            print(f"   ❌ shutil.which('ffmpeg'): Not found")
    except Exception as e:
        print(f"   ❌ shutil.which exception: {e}")
    
    # 4. Test actual chunking operation
    print("\n4. CHUNKING OPERATION TEST:")
    
    audio_file = "2 HOURS of English Conversation Dialogues Listening Practice.mp3"
    if Path(audio_file).exists():
        print(f"   📁 Audio file exists: {audio_file}")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_file = temp_path / "test_chunk.wav"
            
            print(f"   📁 Temp directory: {temp_dir}")
            print(f"   📁 Chunk file target: {chunk_file}")
            
            # Test chunking command
            try:
                cmd = [
                    'ffmpeg',
                    '-i', str(audio_file),
                    '-ss', '0',
                    '-t', '10',
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    str(chunk_file)
                ]
                
                print(f"   🔧 Command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and chunk_file.exists():
                    print(f"   ✅ Chunking test: SUCCESS")
                    print(f"   📊 Chunk file size: {chunk_file.stat().st_size} bytes")
                else:
                    print(f"   ❌ Chunking test failed:")
                    print(f"      Return code: {result.returncode}")
                    print(f"      Stderr: {result.stderr}")
                    print(f"      File exists: {chunk_file.exists()}")
                    
            except Exception as e:
                print(f"   ❌ Chunking test exception: {e}")
    else:
        print(f"   ❌ Audio file not found: {audio_file}")
    
    # 5. Chocolatey FFmpeg suggestion
    print("\n5. INSTALLATION RECOMMENDATIONS:")
    print("   💡 If issues persist, try:")
    print("      choco install ffmpeg")
    print("   💡 Or download from: https://ffmpeg.org/download.html")
    print("   💡 Ensure FFmpeg is in system PATH, not just user PATH")

if __name__ == "__main__":
    diagnose_ffmpeg()
