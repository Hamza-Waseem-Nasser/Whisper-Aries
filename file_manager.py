#!/usr/bin/env python3
"""
File manager for organizing Whisper transcription outputs
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib

class TranscriptionFileManager:
    """Organize and manage transcription files efficiently"""
    
    def __init__(self, base_dir="transcriptions"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized directory structure"""
        directories = [
            "audio_files",           # Original audio files
            "transcriptions/json",   # JSON transcription files
            "transcriptions/txt",    # Plain text transcriptions
            "transcriptions/srt",    # SRT subtitle files
            "performance_logs",      # Performance monitoring data
            "batch_jobs",           # Batch processing results
            "archived",             # Archived old files
            "temp"                  # Temporary files
        ]
        
        for dir_path in directories:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate SHA256 hash for file identification"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # First 16 chars for brevity
    
    def create_session_folder(self, session_name: Optional[str] = None) -> Path:
        """Create a session folder for a transcription batch"""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"
        
        session_dir = self.base_dir / "batch_jobs" / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for the session
        for subdir in ["audio", "transcriptions", "performance", "logs"]:
            (session_dir / subdir).mkdir(exist_ok=True)
        
        return session_dir
    
    def organize_transcription_files(self, audio_file: Path, 
                                   transcription_result: Dict[Any, Any],
                                   performance_data: Optional[Dict] = None,
                                   session_dir: Optional[Path] = None,
                                   copy_audio: bool = False) -> Dict[str, Path]:
        """Organize all files from a transcription job"""
        
        # Use session directory or create organized structure
        if session_dir:
            base_path = session_dir
            audio_dest = base_path / "audio"
            json_dest = base_path / "transcriptions"
            txt_dest = base_path / "transcriptions"
            srt_dest = base_path / "transcriptions"
            perf_dest = base_path / "performance"
        else:
            audio_dest = self.base_dir / "audio_files"
            json_dest = self.base_dir / "transcriptions" / "json"
            txt_dest = self.base_dir / "transcriptions" / "txt"
            srt_dest = self.base_dir / "transcriptions" / "srt"
            perf_dest = self.base_dir / "performance_logs"
        
        # Generate file hash and timestamp
        file_hash = self.get_file_hash(audio_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{audio_file.stem}_{timestamp}_{file_hash}"
        
        organized_files = {}
        
        # 1. Optionally copy audio file (disabled by default for large files)
        if copy_audio:
            audio_copy = audio_dest / f"{base_name}{audio_file.suffix}"
            if not audio_copy.exists():
                shutil.copy2(audio_file, audio_copy)
            organized_files['audio'] = audio_copy
        else:
            # Just reference the original file
            organized_files['audio_original'] = audio_file
        
        # 2. Save JSON transcription
        json_file = json_dest / f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, indent=2, ensure_ascii=False)
        organized_files['json'] = json_file
        
        # 3. Save plain text
        txt_file = txt_dest / f"{base_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(transcription_result.get('text', ''))
        organized_files['txt'] = txt_file
        
        # 4. Save SRT subtitles
        srt_file = srt_dest / f"{base_name}.srt"
        self.save_srt_file(transcription_result, srt_file)
        organized_files['srt'] = srt_file
        
        # 5. Save performance data if available
        if performance_data:
            perf_file = perf_dest / f"{base_name}_performance.json"
            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2)
            organized_files['performance'] = perf_file
        
        # 6. Create metadata file
        metadata_file = json_dest / f"{base_name}_metadata.json"
        metadata = {
            'original_file': str(audio_file),
            'transcription_date': datetime.now().isoformat(),
            'file_hash': file_hash,
            'file_size_mb': audio_file.stat().st_size / (1024 * 1024),
            'duration_seconds': transcription_result.get('duration', 0),
            'language': transcription_result.get('language', 'unknown'),
            'language_probability': transcription_result.get('language_probability', 0),
            'model_used': transcription_result.get('model_size', 'unknown'),
            'processing_time': transcription_result.get('processing_time', 0),
            'speed_ratio': transcription_result.get('speed_ratio', 0),
            'organized_files': {k: str(v) for k, v in organized_files.items()}
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        organized_files['metadata'] = metadata_file
        
        return organized_files
    
    def save_srt_file(self, transcription_result: Dict, srt_file: Path):
        """Save transcription as SRT subtitle file"""
        segments = transcription_result.get('segments', [])
        
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = self.seconds_to_srt_time(segment['start'])
                end_time = self.seconds_to_srt_time(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def create_batch_summary(self, session_dir: Path, batch_results: list) -> Path:
        """Create a summary report for batch processing"""
        summary_file = session_dir / "batch_summary.json"
        
        total_files = len(batch_results)
        successful = len([r for r in batch_results if r.get('success', False)])
        failed = total_files - successful
        
        total_duration = sum(r.get('duration', 0) for r in batch_results if r.get('success', False))
        total_processing_time = sum(r.get('processing_time', 0) for r in batch_results if r.get('success', False))
        
        summary = {
            'session_date': datetime.now().isoformat(),
            'session_folder': str(session_dir),
            'statistics': {
                'total_files': total_files,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total_files * 100) if total_files > 0 else 0,
                'total_audio_duration_minutes': total_duration / 60,
                'total_processing_time_minutes': total_processing_time / 60,
                'average_speed_ratio': sum(r.get('speed_ratio', 0) for r in batch_results if r.get('success', False)) / successful if successful > 0 else 0
            },
            'files_processed': batch_results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary_file
    
    def archive_old_files(self, days_old: int = 30):
        """Archive files older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        archived_count = 0
        
        for folder in ["transcriptions", "performance_logs"]:
            folder_path = self.base_dir / folder
            
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                        # Create archive structure
                        relative_path = file_path.relative_to(self.base_dir)
                        archive_path = self.base_dir / "archived" / relative_path
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move file to archive
                        shutil.move(str(file_path), str(archive_path))
                        archived_count += 1
        
        return archived_count
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = self.base_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir()
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        stats = {}
        
        for folder in ["audio_files", "transcriptions", "performance_logs", "batch_jobs", "archived"]:
            folder_path = self.base_dir / folder
            if folder_path.exists():
                total_size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
                file_count = len(list(folder_path.rglob("*")))
                
                stats[folder] = {
                    'size_mb': total_size / (1024 * 1024),
                    'file_count': file_count
                }
        
        return stats
    
    def find_transcription_by_audio(self, audio_file: Path) -> list:
        """Find all transcriptions for a specific audio file"""
        file_hash = self.get_file_hash(audio_file)
        results = []
        
        # Search in organized transcriptions
        json_dir = self.base_dir / "transcriptions" / "json"
        if json_dir.exists():
            for json_file in json_dir.glob(f"*{file_hash}*.json"):
                if not json_file.name.endswith("_metadata.json"):
                    results.append(json_file)
        
        # Search in batch jobs
        batch_dir = self.base_dir / "batch_jobs"
        if batch_dir.exists():
            for json_file in batch_dir.rglob(f"*{file_hash}*.json"):
                if not json_file.name.endswith("_metadata.json"):
                    results.append(json_file)
        
        return results
    
    def print_storage_summary(self):
        """Print storage usage summary"""
        stats = self.get_storage_stats()
        
        print(f"\n📁 STORAGE SUMMARY")
        print("=" * 40)
        
        total_size = 0
        total_files = 0
        
        for folder, data in stats.items():
            size_mb = data['size_mb']
            file_count = data['file_count']
            total_size += size_mb
            total_files += file_count
            
            print(f"📂 {folder:<15}: {size_mb:8.1f} MB ({file_count:4d} files)")
        
        print("-" * 40)
        print(f"📊 Total Storage    : {total_size:8.1f} MB ({total_files:4d} files)")
        print(f"📍 Base Directory   : {self.base_dir}")


if __name__ == "__main__":
    # Test the file manager
    manager = TranscriptionFileManager()
    manager.print_storage_summary()
    print("File manager initialized successfully!")
