#!/usr/bin/env python3
"""
Performance diagnostic script to identify bottlenecks in parallel transcription
"""

import psutil
import GPUtil
import time
import threading
from pathlib import Path

def monitor_resources(duration=30, interval=1):
    """Monitor system resources during transcription"""
    
    print(f"🔍 MONITORING SYSTEM RESOURCES FOR {duration} SECONDS")
    print("=" * 60)
    
    start_time = time.time()
    measurements = []
    
    while time.time() - start_time < duration:
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU
            gpus = GPUtil.getGPUs()
            gpu_load = gpus[0].load * 100 if gpus else 0
            gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
            gpu_temp = gpus[0].temperature if gpus else 0
            
            # Process count
            python_processes = len([p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()])
            
            measurement = {
                'time': time.time() - start_time,
                'cpu': cpu_percent,
                'memory': memory.percent,
                'gpu_load': gpu_load,
                'gpu_memory': gpu_memory,
                'gpu_temp': gpu_temp,
                'python_processes': python_processes
            }
            
            measurements.append(measurement)
            
            print(f"⚡ CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% | "
                  f"GPU: {gpu_load:5.1f}% | GPU Mem: {gpu_memory:5.1f}% | "
                  f"GPU Temp: {gpu_temp:3.0f}°C | Python Procs: {python_processes}")
            
            time.sleep(interval)
            
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            break
    
    # Analysis
    print(f"\n📊 RESOURCE ANALYSIS:")
    print("=" * 40)
    
    if measurements:
        avg_cpu = sum(m['cpu'] for m in measurements) / len(measurements)
        max_cpu = max(m['cpu'] for m in measurements)
        avg_gpu_load = sum(m['gpu_load'] for m in measurements) / len(measurements)
        max_gpu_load = max(m['gpu_load'] for m in measurements)
        avg_gpu_memory = sum(m['gpu_memory'] for m in measurements) / len(measurements)
        max_gpu_memory = max(m['gpu_memory'] for m in measurements)
        max_python_procs = max(m['python_processes'] for m in measurements)
        
        print(f"🖥️  CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% max")
        print(f"🎮 GPU Load: {avg_gpu_load:.1f}% avg, {max_gpu_load:.1f}% max")
        print(f"🎯 GPU Memory: {avg_gpu_memory:.1f}% avg, {max_gpu_memory:.1f}% max")
        print(f"👥 Max Python Processes: {max_python_procs}")
        
        # Diagnose issues
        print(f"\n🔍 POTENTIAL ISSUES:")
        if max_gpu_memory > 90:
            print("❌ GPU MEMORY SATURATION: >90% usage detected")
        elif max_gpu_memory > 70:
            print("⚠️  GPU MEMORY HIGH: >70% usage detected")
        else:
            print("✅ GPU Memory usage normal")
            
        if max_python_procs > 6:
            print("❌ TOO MANY PYTHON PROCESSES: Possible worker contention")
        elif max_python_procs > 4:
            print("⚠️  HIGH PYTHON PROCESSES: Monitor for contention")
        else:
            print("✅ Python process count normal")
            
        if avg_gpu_load < 50:
            print("⚠️  LOW GPU UTILIZATION: GPU not being used efficiently")
        else:
            print("✅ GPU utilization good")

def check_transcription_processes():
    """Check for running transcription processes"""
    
    print(f"\n🔍 CHECKING TRANSCRIPTION PROCESSES:")
    print("=" * 40)
    
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                
                python_processes.append({
                    'pid': proc.info['pid'],
                    'cmdline': cmdline,
                    'memory_mb': memory_mb
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    print(f"📊 Found {len(python_processes)} Python processes:")
    for i, proc in enumerate(python_processes[:10]):  # Show max 10
        cmdline_short = proc['cmdline'][:80] + "..." if len(proc['cmdline']) > 80 else proc['cmdline']
        print(f"   {i+1}. PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {cmdline_short}")
    
    # Check for transcription-related processes
    transcription_procs = [p for p in python_processes if 'enhanced_chunked' in p['cmdline'] or 'whisper' in p['cmdline'].lower()]
    print(f"\n🎵 Transcription processes: {len(transcription_procs)}")
    
    return python_processes, transcription_procs

if __name__ == "__main__":
    print("🔍 PERFORMANCE DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Check current processes
    python_procs, transcription_procs = check_transcription_processes()
    
    # Monitor for 30 seconds if transcription is running
    if transcription_procs:
        print(f"\n🎵 Transcription detected! Monitoring performance...")
        monitor_resources(duration=30, interval=2)
    else:
        print(f"\n💤 No transcription running. Start enhanced_chunked_demo.py first.")
        print(f"   Then run this script again to monitor performance.")
