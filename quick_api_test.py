"""
quick_api_test.py - Quick test of the API with an existing audio file
"""

import requests
import time
import json
import os

# API base URL
BASE_URL = "http://localhost:8001"

def test_with_harvard_audio():
    """Test the API with the harvard.wav file"""
    print("🧪 Quick API Test with harvard.wav")
    print("=" * 50)
    
    # Check if harvard.wav exists
    audio_file = "harvard.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    print(f"📤 Uploading {audio_file}...")
    
    try:
        # Upload file
        with open(audio_file, "rb") as f:
            files = {"file": f}
            data = {
                "language": "en",
                "confidence_threshold": 0.7,
                "formats": "html,json,srt",
                "run_llm_analysis": False  # Skip LLM since no API key
            }
            
            response = requests.post(f"{BASE_URL}/analyze/", files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            job_id = result['job_id']
            print(f"✅ Upload successful! Job ID: {job_id}")
            
        # Monitor progress
        print("⏳ Monitoring progress...")
        max_wait = 120  # 2 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/status/{job_id}")
            status = status_response.json()
            
            print(f"📊 Status: {status['status']} - {status['message']} ({status['progress']}%)")
            
            if status['status'] == 'completed':
                print("🎉 Analysis completed successfully!")
                
                # Show results
                if 'result' in status and status['result']:
                    stats = status['result']['stats']
                    print(f"📈 Results Summary:")
                    print(f"   - Duration: {stats['total_duration']:.1f}s")
                    print(f"   - Speakers: {stats['num_speakers']}")
                    print(f"   - Segments: {stats['num_segments']}")
                    print(f"   - Available outputs: {list(status['result']['outputs'].keys())}")
                    
                    # Download HTML result
                    try:
                        download_response = requests.get(f"{BASE_URL}/download/{job_id}/html")
                        download_response.raise_for_status()
                        
                        with open("api_test_result.html", "wb") as f:
                            f.write(download_response.content)
                        
                        print("📥 Downloaded HTML transcript: api_test_result.html")
                        
                    except Exception as e:
                        print(f"⚠️ Download failed: {e}")
                
                return True
                
            elif status['status'] == 'failed':
                print(f"❌ Analysis failed: {status.get('error', 'Unknown error')}")
                return False
            
            time.sleep(3)  # Check every 3 seconds
        
        print("⏰ Timeout waiting for completion")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_api_health():
    """Test API health"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print("✅ API is healthy!")
        return True
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        print("Make sure the API server is running: python api_server.py")
        return False

if __name__ == "__main__":
    if test_api_health():
        print()
        test_with_harvard_audio()
