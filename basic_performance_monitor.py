#!/usr/bin/env python3
"""
Basic performance monitor using Windows built-in tools
"""

import time
import json
import threading
import subprocess
import platform
from datetime import datetime
from pathlib import Path
import logging

class BasicPerformanceMonitor:
    """Basic performance monitoring using Windows built-in tools"""
    
    def __init__(self, output_dir="performance_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.data = []
        self.start_time = None
        self.monitor_thread = None
        self.is_windows = platform.system() == "Windows"
        
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
    
    def get_windows_gpu_info(self):
        """Get GPU info using nvidia-smi on Windows"""
        try:
            if not self.is_windows:
                return self._get_default_gpu_info()
            
            # Try to get GPU info using nvidia-smi
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                if len(gpu_data) >= 5:
                    return {
                        'gpu_name': gpu_data[0],
                        'gpu_load': float(gpu_data[1]),
                        'gpu_memory_used': float(gpu_data[2]),
                        'gpu_memory_total': float(gpu_data[3]),
                        'gpu_memory_percent': (float(gpu_data[2]) / float(gpu_data[3])) * 100,
                        'gpu_temperature': float(gpu_data[4])
                    }
            
            return self._get_default_gpu_info()
            
        except Exception as e:
            self.logger.warning(f"GPU monitoring error: {e}")
            return self._get_default_gpu_info()
    
    def _get_default_gpu_info(self):
        """Default GPU info when monitoring fails"""
        return {
            'gpu_name': 'GPU monitoring unavailable',
            'gpu_load': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'gpu_memory_percent': 0,
            'gpu_temperature': 0
        }
    
    def get_windows_system_info(self):
        """Get system info using Windows built-in tools"""
        try:
            # Get CPU usage using wmic
            cpu_result = subprocess.run([
                "wmic", "cpu", "get", "loadpercentage", "/value"
            ], capture_output=True, text=True, timeout=5)
            
            cpu_percent = 0
            if cpu_result.returncode == 0:
                for line in cpu_result.stdout.split('\n'):
                    if 'LoadPercentage' in line:
                        cpu_percent = float(line.split('=')[1].strip())
                        break
            
            # Get memory info using wmic
            memory_result = subprocess.run([
                "wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/value"
            ], capture_output=True, text=True, timeout=5)
            
            total_memory = 0
            free_memory = 0
            
            if memory_result.returncode == 0:
                for line in memory_result.stdout.split('\n'):
                    if 'TotalVisibleMemorySize' in line:
                        total_memory = float(line.split('=')[1].strip()) * 1024  # Convert to bytes
                    elif 'FreePhysicalMemory' in line:
                        free_memory = float(line.split('=')[1].strip()) * 1024  # Convert to bytes
            
            used_memory = total_memory - free_memory
            memory_percent = (used_memory / total_memory * 100) if total_memory > 0 else 0
            
            # Get GPU info
            gpu_info = self.get_windows_gpu_info()
            
            return {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_used_gb': used_memory / (1024**3),
                'memory_total_gb': total_memory / (1024**3),
                'memory_percent': memory_percent,
                **gpu_info
            }
            
        except Exception as e:
            self.logger.error(f"System monitoring error: {e}")
            return {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'cpu_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'memory_percent': 0,
                **self._get_default_gpu_info()
            }
    
    def _monitor_loop(self, interval=3):
        """Main monitoring loop"""
        self.logger.info(f"Starting basic performance monitoring (interval: {interval}s)")
        
        while self.monitoring:
            try:
                system_info = self.get_windows_system_info()
                system_info['elapsed_time'] = time.time() - self.start_time
                self.data.append(system_info)
                
                # Print real-time stats
                print(f"\r⚡ CPU: {system_info['cpu_percent']:5.1f}% | "
                      f"RAM: {system_info['memory_percent']:5.1f}% | "
                      f"GPU: {system_info['gpu_load']:5.1f}% | "
                      f"Time: {system_info['elapsed_time']:6.1f}s", end="")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def start_monitoring(self, interval=3):
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
        
        self.logger.info("Basic performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring:
            self.logger.warning("Monitoring not running")
            return {}
        
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
        
        summary = {
            'monitoring_duration': self.data[-1]['elapsed_time'],
            'data_points': len(self.data),
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'memory_max': max(memory_values),
            'gpu_load_avg': sum(gpu_load_values) / len(gpu_load_values),
            'gpu_load_max': max(gpu_load_values),
            'gpu_name': self.data[0]['gpu_name'],
            'memory_total_gb': self.data[0]['memory_total_gb']
        }
        
        return summary
    
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
        print(f"🔧 GPU: {summary['gpu_name']}")
        print(f"💾 Total RAM: {summary['memory_total_gb']:.1f} GB")
    
    def save_performance_log(self, filename=None):
        """Save performance data to file"""
        if not self.data:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_basic_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'detailed_data': self.data
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance log saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Test the basic performance monitor
    monitor = BasicPerformanceMonitor()
    
    print("Testing basic performance monitor for 10 seconds...")
    monitor.start_monitoring(interval=2)
    
    time.sleep(10)
    
    monitor.stop_monitoring()
    monitor.print_summary()
    monitor.save_performance_log()
