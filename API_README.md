# AI Conversation Analysis API

A production-ready FastAPI server that exposes your advanced conversation analysis pipeline as a REST API. Upload audio/video files and get back speaker-separated transcripts with meeting summaries, action items, and analytics.

## Features

- **🎯 Speaker-aware transcription**: Combines Whisper (transcription) + pyannote.audio (speaker diarization)
- **🧠 LLM meeting analysis**: OpenAI GPT-4o powered summaries and action items
- **🌍 Multilingual support**: Auto-detect or specify language (English, Arabic, etc.)
- **📊 Multiple output formats**: HTML, JSON, SRT, and meeting summaries
- **🔄 Asynchronous processing**: Background job processing with status tracking
- **🚀 Production-ready**: CORS support, error handling, file validation
- **📖 Interactive docs**: Automatic OpenAPI/Swagger documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
# Required for speaker diarization
set HF_TOKEN=your_huggingface_token_here

# Required for LLM analysis
set OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the API Server

```bash
python api_server.py
```

The server will start on `http://localhost:8000`

### 4. View Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 📤 Upload & Analyze

**POST** `/analyze/`

Upload an audio/video file for analysis.

**Parameters:**
- `file` (required): Audio/video file to analyze
- `language` (optional): Language code (default: "auto")
- `confidence_threshold` (optional): Speaker confidence threshold 0.0-1.0 (default: 0.7)
- `formats` (optional): Output formats, comma-separated (default: "html,json,srt")
- `run_llm_analysis` (optional): Whether to run LLM analysis (default: true)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/analyze/" \
  -F "file=@meeting.mp4" \
  -F "language=en" \
  -F "confidence_threshold=0.8" \
  -F "formats=html,json,srt" \
  -F "run_llm_analysis=true"
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "File uploaded successfully. Processing started.",
  "filename": "meeting.mp4"
}
```

### 📊 Check Job Status

**GET** `/status/{job_id}`

Get the current status of a processing job.

**Example:**
```bash
curl http://localhost:8000/status/123e4567-e89b-12d3-a456-426614174000
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 100,
  "message": "Analysis completed successfully",
  "created_at": "2025-07-10T10:30:00",
  "completed_at": "2025-07-10T10:33:45",
  "result": {
    "success": true,
    "outputs": {
      "html": "api_outputs/123e4567.../meeting.html",
      "json": "api_outputs/123e4567.../meeting.json",
      "srt": "api_outputs/123e4567.../meeting.srt",
      "meeting_summary_txt": "api_outputs/123e4567.../meeting.meeting_summary.txt",
      "meeting_summary_html": "api_outputs/123e4567.../meeting.meeting_summary.html"
    },
    "stats": {
      "total_duration": 1847.5,
      "num_speakers": 3,
      "num_segments": 245
    }
  }
}
```

### 📥 Download Results

**GET** `/download/{job_id}/{file_type}`

Download a specific result file.

**File Types:**
- `html`: Interactive HTML transcript with speaker colors
- `json`: Structured JSON with all data
- `srt`: Standard subtitle format
- `meeting_summary_txt`: Plain text meeting summary
- `meeting_summary_html`: HTML meeting summary

**Example:**
```bash
curl http://localhost:8000/download/123e4567.../html -o meeting_transcript.html
```

### 📋 List Jobs

**GET** `/jobs/`

List all processing jobs (last 50).

### 🗑️ Delete Job

**DELETE** `/jobs/{job_id}`

Delete a job and clean up its output files.

### 📈 API Statistics

**GET** `/stats/`

Get API usage statistics.

## Using with Postman

1. **Create a new request** in Postman
2. **Set method to POST** and URL to `http://localhost:8000/analyze/`
3. **Go to Body tab** → select "form-data"
4. **Add key-value pairs:**
   - `file`: (select file) → choose your audio/video file
   - `language`: text → `en` or `auto`
   - `confidence_threshold`: text → `0.8`
   - `formats`: text → `html,json,srt`
   - `run_llm_analysis`: text → `true`
5. **Send the request**
6. **Copy the job_id** from the response
7. **Create a new GET request** to `http://localhost:8000/status/{job_id}` to check status
8. **When completed, create GET requests** to download results

## Programming Examples

### Python with requests

```python
import requests
import time

# Upload file
with open("meeting.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/",
        files={"file": f},
        data={
            "language": "en",
            "confidence_threshold": 0.8,
            "formats": "html,json,srt",
            "run_llm_analysis": True
        }
    )

job_id = response.json()["job_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/status/{job_id}").json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Job failed: {status['error']}")
        break
    time.sleep(5)

# Download results
html_response = requests.get(f"http://localhost:8000/download/{job_id}/html")
with open("transcript.html", "wb") as f:
    f.write(html_response.content)
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function analyzeAudio(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('language', 'en');
    form.append('confidence_threshold', '0.8');
    form.append('formats', 'html,json,srt');
    form.append('run_llm_analysis', 'true');
    
    // Upload
    const uploadResponse = await axios.post('http://localhost:8000/analyze/', form, {
        headers: form.getHeaders()
    });
    
    const jobId = uploadResponse.data.job_id;
    
    // Poll for completion
    while (true) {
        const statusResponse = await axios.get(`http://localhost:8000/status/${jobId}`);
        const status = statusResponse.data;
        
        if (status.status === 'completed') {
            console.log('Analysis completed!');
            break;
        } else if (status.status === 'failed') {
            console.error('Analysis failed:', status.error);
            return;
        }
        
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    // Download HTML result
    const downloadResponse = await axios.get(`http://localhost:8000/download/${jobId}/html`, {
        responseType: 'stream'
    });
    
    downloadResponse.data.pipe(fs.createWriteStream('transcript.html'));
}
```

## Testing

Run the included test script to verify everything works:

```bash
python test_api.py
```

This will:
1. Check API health
2. List existing jobs
3. Prompt you to upload a file
4. Monitor processing progress
5. Offer to download results

## Supported File Types

- **Audio**: MP3, WAV, M4A, FLAC
- **Video**: MP4, AVI, MOV, MKV, WEBM (audio will be extracted)

## Configuration

Edit `config.py` to customize:
- Default output directory
- Model settings
- Confidence thresholds
- Language settings

## Production Deployment

For production use:

1. **Use a proper ASGI server:**
   ```bash
   pip install gunicorn
   gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Set up proper authentication and rate limiting**

3. **Use external storage** (S3, etc.) for large files

4. **Use Redis or a database** for job storage instead of in-memory

5. **Configure CORS** appropriately for your domain

6. **Set up monitoring and logging**

## Troubleshooting

### Common Issues

1. **HuggingFace token issues**: Make sure you've accepted the user conditions at https://hf.co/pyannote/speaker-diarization-3.1

2. **OpenAI API key issues**: Verify your API key is set correctly and has credits

3. **GPU memory issues**: Reduce model size or chunk length in the configuration

4. **File format issues**: Use ffmpeg to convert unsupported formats

### Getting Help

- Check the API logs for detailed error messages
- Use the interactive documentation at `/docs`
- Run `python test_api.py` to verify setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.
