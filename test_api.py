"""
test_api.py - Demo script to test the API server

This script demonstrates how to use the API programmatically.
You can also use Postman or curl to test the endpoints.
"""

import requests
import time
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8001"

def test_api_health():
    """Test API health check"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print("✅ API is running!")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def upload_file_for_analysis(file_path, language="auto", confidence_threshold=0.7, formats="html,json,srt"):
    """Upload a file for analysis"""
    print(f"📤 Uploading file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "language": language,
                "confidence_threshold": confidence_threshold,
                "formats": formats,
                "run_llm_analysis": True
            }
            
            response = requests.post(f"{BASE_URL}/analyze/", files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ File uploaded successfully!")
            print(f"Job ID: {result['job_id']}")
            return result['job_id']
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def check_job_status(job_id):
    """Check the status of a job"""
    try:
        response = requests.get(f"{BASE_URL}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return None

def wait_for_job_completion(job_id, timeout=300):
    """Wait for job to complete"""
    print(f"⏳ Waiting for job {job_id} to complete...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = check_job_status(job_id)
        if not status:
            return False
        
        print(f"Status: {status['status']} - {status['message']} ({status['progress']}%)")
        
        if status['status'] == 'completed':
            print("✅ Job completed successfully!")
            return True
        elif status['status'] == 'failed':
            print(f"❌ Job failed: {status.get('error', 'Unknown error')}")
            return False
        
        time.sleep(5)  # Check every 5 seconds
    
    print(f"⏰ Job timed out after {timeout} seconds")
    return False

def download_result(job_id, file_type, save_path=None):
    """Download a result file"""
    print(f"📥 Downloading {file_type} for job {job_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/download/{job_id}/{file_type}")
        response.raise_for_status()
        
        if save_path is None:
            save_path = f"downloaded_{job_id}_{file_type}"
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def list_all_jobs():
    """List all jobs"""
    print("📋 Listing all jobs...")
    try:
        response = requests.get(f"{BASE_URL}/jobs/")
        response.raise_for_status()
        
        jobs = response.json()['jobs']
        print(f"Found {len(jobs)} jobs:")
        for job in jobs[:5]:  # Show first 5
            print(f"  - {job['job_id']}: {job['status']} ({job['created_at']})")
        
        return jobs
        
    except Exception as e:
        print(f"❌ Failed to list jobs: {e}")
        return []

def main():
    """Main demo function"""
    print("🧪 AI Conversation Analysis API Demo")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("❌ API is not running. Please start the server first:")
        print("   python api_server.py")
        return
    
    print()
    
    # List existing jobs
    list_all_jobs()
    print()
    
    # Ask user for file to upload
    print("📁 Please provide a path to an audio/video file to test:")
    print("   (or press Enter to skip upload test)")
    file_path = input("File path: ").strip()
    
    if not file_path:
        print("⏭️  Skipping upload test")
        return
    
    # Upload file
    job_id = upload_file_for_analysis(file_path)
    if not job_id:
        return
    
    print()
    
    # Wait for completion
    if wait_for_job_completion(job_id):
        print()
        
        # Get final status
        final_status = check_job_status(job_id)
        if final_status and final_status.get('result'):
            result = final_status['result']
            print(f"📊 Results summary:")
            print(f"   - Total duration: {result['stats']['total_duration']:.1f}s")
            print(f"   - Speakers: {result['stats']['num_speakers']}")
            print(f"   - Segments: {result['stats']['num_segments']}")
            print(f"   - Available outputs: {list(result['outputs'].keys())}")
            
            # Ask user what to download
            print("\n📥 What would you like to download?")
            available_types = list(result['outputs'].keys())
            for i, file_type in enumerate(available_types):
                print(f"   {i+1}. {file_type}")
            
            choice = input("Enter number (or 'all' for everything): ").strip()
            
            if choice.lower() == 'all':
                for file_type in available_types:
                    download_result(job_id, file_type)
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(available_types):
                        download_result(job_id, available_types[index])
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Invalid input")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    main()
