#!/usr/bin/env python3
"""
Performance monitoring system for Whisper transcription
"""

import psutil
import time
import json
import threading
import os
from datetime import datetime
from pathlib import Path
import GPUtil
import logging

class PerformanceMonitor:
    """Monitor system performance during transcription"""
    
    def __init__(self, output_dir="performance_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.data = []
        self.start_time = None
        self.monitor_thread = None
        
        # Setup logging
        log_file = self.output_dir / "performance.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming single GPU
                return {
                    'gpu_name': gpu.name,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_load': gpu.load * 100,
                    'gpu_temperature': gpu.temperature
                }
            else:
                return {
                    'gpu_name': 'No GPU detected',
                    'gpu_memory_used': 0,
                    'gpu_memory_total': 0,
                    'gpu_memory_percent': 0,
                    'gpu_load': 0,
                    'gpu_temperature': 0
                }
        except Exception as e:
            self.logger.warning(f"GPU monitoring error: {e}")
            return {
                'gpu_name': 'GPU monitoring failed',
                'gpu_memory_used': 0,
                'gpu_memory_total': 0,
                'gpu_memory_percent': 0,
                'gpu_load': 0,
                'gpu_temperature': 0
            }
    
    def get_system_info(self):
        """Get current system performance metrics"""
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        # Process information
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # GPU information
        gpu_info = self.get_gpu_info()
        
        return {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'cpu_count': cpu_count,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'process_memory_mb': process_memory.rss / (1024**2),
            'process_memory_vms_mb': process_memory.vms / (1024**2),
            **gpu_info
        }
    
    def _monitor_loop(self, interval=2):
        """Main monitoring loop"""
        self.logger.info(f"Starting performance monitoring (interval: {interval}s)")
        
        while self.monitoring:
            try:
                system_info = self.get_system_info()
                system_info['elapsed_time'] = time.time() - self.start_time
                self.data.append(system_info)
                
                # Print real-time stats
                print(f"\r⚡ CPU: {system_info['cpu_percent']:5.1f}% | "
                      f"RAM: {system_info['memory_percent']:5.1f}% | "
                      f"GPU: {system_info['gpu_load']:5.1f}% | "
                      f"GPU Mem: {system_info['gpu_memory_percent']:5.1f}% | "
                      f"Time: {system_info['elapsed_time']:6.1f}s", end="")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def start_monitoring(self, interval=2):
        """Start performance monitoring"""
        if self.monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.data = []
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring:
            self.logger.warning("Monitoring not running")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        print()  # New line after real-time stats
        self.logger.info("Performance monitoring stopped")
        
        return self.get_summary()
    
    def get_summary(self):
        """Get performance summary"""
        if not self.data:
            return {}
        
        # Calculate averages and peaks
        cpu_values = [d['cpu_percent'] for d in self.data]
        memory_values = [d['memory_percent'] for d in self.data]
        gpu_load_values = [d['gpu_load'] for d in self.data]
        gpu_memory_values = [d['gpu_memory_percent'] for d in self.data]
        
        summary = {
            'monitoring_duration': self.data[-1]['elapsed_time'],
            'data_points': len(self.data),
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'memory_max': max(memory_values),
            'gpu_load_avg': sum(gpu_load_values) / len(gpu_load_values),
            'gpu_load_max': max(gpu_load_values),
            'gpu_memory_avg': sum(gpu_memory_values) / len(gpu_memory_values),
            'gpu_memory_max': max(gpu_memory_values),
            'gpu_name': self.data[0]['gpu_name'],
            'system_info': {
                'cpu_count': self.data[0]['cpu_count'],
                'memory_total_gb': self.data[0]['memory_total_gb'],
                'gpu_memory_total': self.data[0]['gpu_memory_total']
            }
        }
        
        return summary
    
    def save_detailed_log(self, filename=None):
        """Save detailed performance log"""
        if not self.data:
            self.logger.warning("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_detailed_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        self.logger.info(f"Detailed performance log saved to {filepath}")
        return filepath
    
    def save_summary_report(self, filename=None, audio_file=None, transcription_result=None):
        """Save performance summary report"""
        summary = self.get_summary()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_summary_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Enhanced summary with transcription info
        report = {
            'timestamp': datetime.now().isoformat(),
            'audio_file': str(audio_file) if audio_file else None,
            'performance_summary': summary,
            'transcription_info': transcription_result,
            'recommendations': self._generate_recommendations(summary)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance summary saved to {filepath}")
        return filepath
    
    def _generate_recommendations(self, summary):
        """Generate performance recommendations"""
        recommendations = []
        
        if summary['cpu_avg'] > 80:
            recommendations.append("High CPU usage detected. Consider using a smaller model or int8 quantization.")
        
        if summary['memory_avg'] > 85:
            recommendations.append("High memory usage detected. Consider closing other applications.")
        
        if summary['gpu_load_avg'] > 90:
            recommendations.append("High GPU load detected. Performance is likely optimal.")
        elif summary['gpu_load_avg'] < 30:
            recommendations.append("Low GPU utilization. Consider using a larger model or fp16 precision.")
        
        if summary['gpu_memory_avg'] > 90:
            recommendations.append("High GPU memory usage. Consider using int8 quantization or a smaller model.")
        
        if not recommendations:
            recommendations.append("Performance looks good! System is well-balanced for transcription.")
        
        return recommendations
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_summary()
        
        if not summary:
            print("❌ No performance data available")
            return
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"🕐 Duration: {summary['monitoring_duration']:.1f} seconds")
        print(f"📈 Data Points: {summary['data_points']}")
        print(f"🖥️  CPU: {summary['cpu_avg']:.1f}% avg, {summary['cpu_max']:.1f}% max")
        print(f"💾 RAM: {summary['memory_avg']:.1f}% avg, {summary['memory_max']:.1f}% max")
        print(f"🎮 GPU: {summary['gpu_load_avg']:.1f}% avg, {summary['gpu_load_max']:.1f}% max")
        print(f"🎯 GPU Memory: {summary['gpu_memory_avg']:.1f}% avg, {summary['gpu_memory_max']:.1f}% max")
        print(f"🔧 GPU: {summary['gpu_name']}")
        
        # Show recommendations
        recommendations = self._generate_recommendations(summary)
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")


if __name__ == "__main__":
    # Test the performance monitor
    monitor = PerformanceMonitor()
    
    print("Testing performance monitor for 10 seconds...")
    monitor.start_monitoring(interval=1)
    
    time.sleep(10)
    
    monitor.stop_monitoring()
    monitor.print_summary()
    monitor.save_detailed_log()
    monitor.save_summary_report()
