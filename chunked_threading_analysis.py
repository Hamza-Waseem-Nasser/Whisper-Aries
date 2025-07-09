#!/usr/bin/env python3
"""
Real-world chunked transcription analysis:
Why chunked threading might be slower despite threading working
"""

import os
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

def create_test_chunks(audio_file, num_chunks=4, chunk_duration=10):
    """Create multiple audio chunks to simulate real chunked processing"""
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
    
    print(f"🔧 Creating {num_chunks} test chunks of {chunk_duration}s each...")
    
    # Find ffmpeg
    ffmpeg_path = None
    for path in [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\ProgramData\chocolatey\bin\ffmpeg.exe", "ffmpeg"]:
        try:
            result = subprocess.run([path, "-version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_path = path
                break
        except:
            continue
    
    if not ffmpeg_path:
        print("❌ FFmpeg not found")
        return []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_file = os.path.join(temp_dir, f"chunk_{i+1}.mp3")
        
        cmd = [
            ffmpeg_path, '-i', audio_file,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-acodec', 'mp3',
            '-y', chunk_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and os.path.exists(chunk_file):
                chunks.append(chunk_file)
                print(f"✅ Created chunk {i+1}: {chunk_file}")
            else:
                print(f"❌ Failed to create chunk {i+1}")
        except Exception as e:
            print(f"❌ Error creating chunk {i+1}: {e}")
    
    return chunks

def transcribe_single_chunk(transcriber, chunk_file, chunk_id):
    """Transcribe a single chunk"""
    thread_id = threading.get_ident()
    print(f"🎯 Transcribing chunk {chunk_id} on thread {thread_id}")
    
    start_time = time.time()
    
    try:
        result = transcriber.transcribe_file(chunk_file)
        processing_time = time.time() - start_time
        
        return {
            'chunk_id': chunk_id,
            'thread_id': thread_id,
            'processing_time': processing_time,
            'success': True,
            'text_length': len(result.get('text', '')),
            'duration': result.get('duration', 0),
            'speed_ratio': result.get('duration', 0) / processing_time if processing_time > 0 else 0
        }
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Chunk {chunk_id} failed: {e}")
        return {
            'chunk_id': chunk_id,
            'thread_id': thread_id,
            'processing_time': processing_time,
            'success': False,
            'error': str(e)
        }

def test_sequential_chunked(transcriber, chunks):
    """Test sequential processing of chunks"""
    print(f"\n🔥 Sequential Chunked Processing ({len(chunks)} chunks)")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    for i, chunk in enumerate(chunks):
        result = transcribe_single_chunk(transcriber, chunk, i + 1)
        results.append(result)
        
        if result['success']:
            print(f"✅ Chunk {i+1}: {result['processing_time']:.2f}s ({result['speed_ratio']:.1f}x real-time)")
        else:
            print(f"❌ Chunk {i+1}: Failed")
    
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    
    print(f"\n📊 Sequential Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful chunks: {len(successful)}/{len(chunks)}")
    if successful:
        avg_speed = sum(r['speed_ratio'] for r in successful) / len(successful)
        print(f"   Average speed ratio: {avg_speed:.1f}x real-time")
    
    return results, total_time

def test_threaded_chunked(transcriber, chunks, max_workers=2):
    """Test threaded processing of chunks"""
    print(f"\n🧵 Threaded Chunked Processing ({len(chunks)} chunks, {max_workers} workers)")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(transcribe_single_chunk, transcriber, chunk, i + 1): i + 1
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✅ Chunk {chunk_id}: {result['processing_time']:.2f}s ({result['speed_ratio']:.1f}x real-time) [thread {result['thread_id']}]")
                else:
                    print(f"❌ Chunk {chunk_id}: Failed")
            except Exception as e:
                print(f"❌ Chunk {chunk_id} exception: {e}")
    
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    
    print(f"\n📊 Threaded Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful chunks: {len(successful)}/{len(chunks)}")
    if successful:
        avg_speed = sum(r['speed_ratio'] for r in successful) / len(successful)
        print(f"   Average speed ratio: {avg_speed:.1f}x real-time")
        
        # Check thread distribution
        thread_ids = set(r['thread_id'] for r in successful)
        print(f"   Threads used: {len(thread_ids)}")
    
    return results, total_time

def analyze_chunked_performance(seq_results, thread_results, seq_time, thread_time):
    """Analyze the difference between sequential and threaded chunked processing"""
    print(f"\n🔍 Chunked Processing Analysis")
    print("=" * 60)
    
    seq_successful = [r for r in seq_results if r['success']]
    thread_successful = [r for r in thread_results if r['success']]
    
    if seq_successful and thread_successful:
        seq_avg_speed = sum(r['speed_ratio'] for r in seq_successful) / len(seq_successful)
        thread_avg_speed = sum(r['speed_ratio'] for r in thread_successful) / len(thread_successful)
        
        print(f"📈 Processing Speed:")
        print(f"   Sequential: {seq_avg_speed:.1f}x real-time per chunk")
        print(f"   Threaded:   {thread_avg_speed:.1f}x real-time per chunk")
        print(f"   Speed degradation: {((thread_avg_speed - seq_avg_speed) / seq_avg_speed * 100):+.1f}%")
        
        print(f"\n⏱️  Wall Clock Time:")
        print(f"   Sequential: {seq_time:.2f}s")
        print(f"   Threaded:   {thread_time:.2f}s")
        print(f"   Time improvement: {((seq_time - thread_time) / seq_time * 100):+.1f}%")
        
        # Calculate theoretical vs actual speedup
        expected_speedup = min(2, len(seq_successful))  # max 2 workers
        actual_speedup = seq_time / thread_time if thread_time > 0 else 0
        efficiency = (actual_speedup / expected_speedup) * 100 if expected_speedup > 0 else 0
        
        print(f"\n🎯 Threading Efficiency:")
        print(f"   Expected speedup: {expected_speedup:.1f}x")
        print(f"   Actual speedup: {actual_speedup:.1f}x")
        print(f"   Efficiency: {efficiency:.1f}%")
        
        if efficiency < 50:
            print(f"\n⚠️  Low threading efficiency detected!")
            print(f"📝 Possible causes:")
            print(f"   • GPU context switching overhead")
            print(f"   • Model sharing conflicts")
            print(f"   • CUDA stream serialization")
            print(f"   • Memory bandwidth bottleneck")
            print(f"   • Chunk size too small for effective parallelization")

def main():
    """Main analysis function"""
    print("🚀 Real-World Chunked Transcription Analysis")
    print("=" * 60)
    
    # Load transcriber
    from transcribe import WhisperTranscriber
    print("🔄 Loading Whisper model...")
    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        languages=["en"]
    )
    
    # Create test chunks from example audio
    audio_file = r"c:\Users\ASUS\Whisper Project\example.mp3"
    if not os.path.exists(audio_file):
        print(f"❌ Test audio file not found: {audio_file}")
        return
    
    chunks = create_test_chunks(audio_file, num_chunks=4, chunk_duration=10)
    if not chunks:
        print("❌ Failed to create test chunks")
        return
    
    try:
        # Test sequential
        seq_results, seq_time = test_sequential_chunked(transcriber, chunks)
        
        # Small delay
        time.sleep(2)
        
        # Test threaded
        thread_results, thread_time = test_threaded_chunked(transcriber, chunks, max_workers=2)
        
        # Analyze
        analyze_chunked_performance(seq_results, thread_results, seq_time, thread_time)
        
    finally:
        # Cleanup chunks
        for chunk in chunks:
            try:
                if os.path.exists(chunk):
                    os.remove(chunk)
            except:
                pass
        
        # Remove temp directory
        try:
            import shutil
            temp_dir = os.path.dirname(chunks[0]) if chunks else None
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    main()
