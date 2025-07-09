#!/usr/bin/env python3
"""
GPU-Optimized Sequential Chunked Transcriber
Based on performance analysis showing threading is slower for GPU workloads
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import shutil

class GPUOptimizedTranscriber:
    """GPU-optimized sequential chunked transcription"""
    
    def __init__(self, model_size="large-v3", device="cuda", compute_type="int8", 
                 languages=None, chunk_length_minutes=5, overlap_seconds=10):
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.languages = languages or ["ar", "en"]
        self.chunk_length_minutes = chunk_length_minutes
        self.overlap_seconds = overlap_seconds
        
        # Sequential processing for optimal GPU performance
        self.transcriber = None
        
        print(f"🚀 GPU-Optimized Sequential Transcriber Config:")
        print(f"   📏 Chunk length: {chunk_length_minutes} minutes")
        print(f"   🔄 Overlap: {overlap_seconds} seconds")
        print(f"   🎮 Device: {device} (sequential processing)")
        print(f"   🧠 Model: {model_size}")
        print(f"   ⚡ Mode: Sequential (optimal for GPU)")
    
    def get_audio_duration_ffprobe(self, audio_path):
        """Get audio duration using ffprobe"""
        try:
            # Find ffprobe
            ffprobe_path = None
            for path in [r"C:\ffmpeg\bin\ffprobe.exe", r"C:\ProgramData\chocolatey\bin\ffprobe.exe", "ffprobe"]:
                try:
                    result = subprocess.run([path, "-version"], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        ffprobe_path = path
                        break
                except:
                    continue
            
            if not ffprobe_path:
                raise Exception("ffprobe not found")
            
            cmd = [
                ffprobe_path, '-v', 'quiet', '-print_format', 'json',
                '-show_format', str(Path(audio_path).absolute())
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                print(f"📏 Audio duration: {duration/60:.1f} minutes")
                return duration
            else:
                raise Exception(f"ffprobe failed: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️  ffprobe failed: {e}")
            # Fallback to transcriber duration detection
            return None
    
    def create_audio_chunk(self, audio_path, start_time, end_time, chunk_index, temp_dir):
        """Create audio chunk using ffmpeg"""
        try:
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
                raise Exception("ffmpeg not found")
            
            chunk_file = Path(temp_dir) / f"chunk_{chunk_index:03d}.wav"
            
            cmd = [
                ffmpeg_path,
                '-i', str(Path(audio_path).absolute()),
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(chunk_file.absolute())
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and chunk_file.exists():
                print(f"✅ Created chunk {chunk_index + 1}: {start_time/60:.1f}-{end_time/60:.1f} min")
                return chunk_file
            else:
                print(f"❌ Failed to create chunk {chunk_index + 1}: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating chunk {chunk_index + 1}: {e}")
            return None
    
    def get_transcriber(self):
        """Get transcriber instance (load once, reuse)"""
        if self.transcriber is None:
            print("🔄 Loading Whisper model...")
            start_time = time.time()
            
            from transcribe import WhisperTranscriber
            self.transcriber = WhisperTranscriber(
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                languages=self.languages
            )
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
        return self.transcriber
    
    def transcribe_chunk_sequential(self, chunk_file, start_time, end_time, chunk_index):
        """Transcribe a single chunk sequentially"""
        if chunk_file is None:
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': False,
                'error': 'Chunk file creation failed'
            }
        
        try:
            print(f"🎯 Processing chunk {chunk_index + 1} ({start_time/60:.1f}-{end_time/60:.1f} min)")
            
            chunk_start_time = time.time()
            transcriber = self.get_transcriber()
            result = transcriber.transcribe_file(str(chunk_file))
            
            processing_time = time.time() - chunk_start_time
            audio_duration = result.get('duration', end_time - start_time)
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            print(f"✅ Chunk {chunk_index + 1}: {processing_time:.2f}s ({speed_ratio:.1f}x real-time)")
            
            # Adjust segment timestamps to global timeline
            if 'segments' in result:
                for segment in result['segments']:
                    segment['start'] += start_time
                    segment['end'] += start_time
            
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'speed_ratio': speed_ratio
            }
            
        except Exception as e:
            processing_time = time.time() - chunk_start_time if 'chunk_start_time' in locals() else 0
            print(f"❌ Chunk {chunk_index + 1} failed: {e}")
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
        finally:
            # Clean up chunk file
            try:
                if chunk_file and os.path.exists(chunk_file):
                    os.remove(chunk_file)
            except:
                pass
    
    def remove_overlapping_segments(self, segments):
        """Remove duplicate segments at chunk boundaries"""
        if len(segments) <= 1:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        filtered_segments = [segments[0]]
        
        for segment in segments[1:]:
            last_segment = filtered_segments[-1]
            
            # Check for overlap (within 1 second tolerance)
            if segment['start'] <= last_segment['end'] + 1.0:
                # Overlapping segments - keep the longer one or merge
                if segment['end'] > last_segment['end']:
                    # Current segment extends further, replace
                    filtered_segments[-1] = segment
                # Otherwise skip the current segment
            else:
                # No overlap, add the segment
                filtered_segments.append(segment)
        
        return filtered_segments
    
    def merge_results(self, chunk_results, total_duration):
        """Merge chunk results into final transcription"""
        print("🔄 Merging chunk results...")
        
        # Sort by chunk index
        chunk_results.sort(key=lambda x: x['chunk_index'])
        
        # Collect all segments
        all_segments = []
        total_processing_time = 0
        successful_chunks = 0
        
        for chunk in chunk_results:
            if chunk['success']:
                successful_chunks += 1
                total_processing_time += chunk.get('processing_time', 0)
                
                result = chunk['result']
                if 'segments' in result:
                    all_segments.extend(result['segments'])
        
        # Remove duplicate segments at chunk boundaries
        all_segments = self.remove_overlapping_segments(all_segments)
        
        # Build full text
        full_text = " ".join(segment['text'].strip() for segment in all_segments)
        
        # Get language from first successful chunk
        language = "en"
        language_probability = 1.0
        
        for chunk in chunk_results:
            if chunk['success'] and 'result' in chunk:
                language = chunk['result'].get('language', 'en')
                language_probability = chunk['result'].get('language_probability', 1.0)
                break
        
        merged_result = {
            'text': full_text,
            'segments': all_segments,
            'language': language,
            'language_probability': language_probability,
            'duration': total_duration,
            'processing_time': total_processing_time,
            'speed_ratio': total_duration / total_processing_time if total_processing_time > 0 else 0,
            'chunks_processed': successful_chunks,
            'chunks_total': len(chunk_results),
            'chunked_transcription': True,
            'processing_mode': 'sequential_gpu_optimized'
        }
        
        print(f"✅ Merged {successful_chunks}/{len(chunk_results)} chunks")
        print(f"🚀 Overall speed: {merged_result['speed_ratio']:.1f}x real-time")
        
        return merged_result
    
    def transcribe_file(self, audio_path, output_formats=None):
        """Transcribe audio file using GPU-optimized sequential chunked processing"""
        if output_formats is None:
            output_formats = ['json', 'txt', 'srt']
        
        audio_path = Path(audio_path).absolute()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"🎵 GPU-Optimized Sequential Transcription: {audio_path.name}")
        print("=" * 60)
        
        # Get audio duration
        total_duration = self.get_audio_duration_ffprobe(audio_path)
        if total_duration is None:
            print("⚠️  Could not determine duration, will process as single chunk")
            # Set a reasonable default for single file processing
            total_duration = 300  # 5 minutes default
        
        if total_duration <= 0:
            raise ValueError("Could not determine audio duration")
        
        print(f"📏 Total duration: {total_duration/60:.1f} minutes")
        
        # Calculate chunks
        chunk_length = self.chunk_length_minutes * 60  # Convert to seconds
        overlap = self.overlap_seconds
        
        chunks_info = []
        start_time = 0
        chunk_index = 0
        
        while start_time < total_duration:
            end_time = min(start_time + chunk_length, total_duration)
            chunks_info.append((start_time, end_time, chunk_index))
            start_time = end_time - overlap  # Overlap for better accuracy
            chunk_index += 1
            
            if end_time >= total_duration:
                break
        
        print(f"📦 Created {len(chunks_info)} chunks")
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory(prefix="whisper_gpu_chunks_") as temp_dir:
            print(f"📁 Working directory: {temp_dir}")
            
            # Process chunks sequentially
            overall_start_time = time.time()
            chunk_results = []
            
            for chunk_index, (start_time, end_time, idx) in enumerate(chunks_info):
                # Create chunk
                chunk_file = self.create_audio_chunk(audio_path, start_time, end_time, chunk_index, temp_dir)
                
                # Transcribe chunk
                result = self.transcribe_chunk_sequential(chunk_file, start_time, end_time, chunk_index)
                chunk_results.append(result)
            
            # Merge results
            final_result = self.merge_results(chunk_results, total_duration)
            
            overall_time = time.time() - overall_start_time
            print(f"\n🎯 Final Results:")
            print(f"   Overall processing time: {overall_time:.2f}s")
            print(f"   Overall speed ratio: {final_result['speed_ratio']:.1f}x real-time")
            print(f"   Mode: Sequential GPU-optimized")
        
        # Save outputs
        base_name = audio_path.stem
        output_dir = audio_path.parent
        
        saved_files = []
        
        if 'json' in output_formats:
            json_file = output_dir / f"{base_name}_transcription.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            saved_files.append(json_file)
            print(f"💾 Saved: {json_file}")
        
        if 'txt' in output_formats:
            txt_file = output_dir / f"{base_name}_transcription.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(final_result['text'])
            saved_files.append(txt_file)
            print(f"💾 Saved: {txt_file}")
        
        if 'srt' in output_formats:
            srt_file = output_dir / f"{base_name}_transcription.srt"
            self.save_srt(final_result['segments'], srt_file)
            saved_files.append(srt_file)
            print(f"💾 Saved: {srt_file}")
        
        return final_result
    
    def save_srt(self, segments, output_file):
        """Save segments as SRT subtitle file"""
        def format_timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millisecs = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

def main():
    """Demo of GPU-optimized sequential transcriber"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gpu_optimized_transcriber.py <audio_file>")
        print("Example: python gpu_optimized_transcriber.py example.mp3")
        return
    
    audio_file = sys.argv[1]
    
    # Create transcriber
    transcriber = GPUOptimizedTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        languages=["ar", "en"],
        chunk_length_minutes=5,
        overlap_seconds=10
    )
    
    try:
        # Transcribe
        result = transcriber.transcribe_file(audio_file)
        
        print(f"\n🎉 Transcription completed!")
        print(f"   Language: {result['language']} ({result['language_probability']*100:.1f}%)")
        print(f"   Duration: {result['duration']/60:.1f} minutes")
        print(f"   Speed: {result['speed_ratio']:.1f}x real-time")
        print(f"   Text length: {len(result['text'])} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
