"""
api_server.py - FastAPI server for conversation analysis pipeline

Exposes the conversation analysis pipeline as a REST API for:
- Single file upload and analysis
- Batch processing
- Status checking
- Result downloading

Usage:
    python api_server.py
    
Then use Postman or curl to interact with the API endpoints.
"""

import os
import shutil
import tempfile
import uuid
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your pipeline
from conversation_transcriber import run_pipeline
import config

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI Conversation Analysis API",
    description="Speaker-aware multilingual transcription with meeting analytics",
    version="1.0.0"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Job Management ---
JOBS_FILE = "api_jobs.pkl"

def load_jobs():
    """Load jobs from persistent storage"""
    try:
        if os.path.exists(JOBS_FILE):
            with open(JOBS_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Could not load jobs from {JOBS_FILE}: {e}")
    return {}

def save_jobs(jobs_dict):
    """Save jobs to persistent storage"""
    try:
        with open(JOBS_FILE, 'wb') as f:
            pickle.dump(jobs_dict, f)
    except Exception as e:
        print(f"⚠️ Could not save jobs to {JOBS_FILE}: {e}")

jobs = load_jobs()  # Load existing jobs on startup
executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent jobs

class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    progress: int  # 0-100
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalysisRequest(BaseModel):
    language: str = "auto"
    confidence_threshold: float = 0.7
    formats: List[str] = ["html", "json", "srt"]
    run_llm_analysis: bool = True

# --- Helper Functions ---
def create_job(filename: str) -> str:
    """Create a new job and return job ID"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0,
        message="Job created",
        created_at=datetime.now().isoformat()
    )
    save_jobs(jobs)  # Persist to disk
    return job_id

def update_job_status(job_id: str, status: str, progress: int = None, message: str = None, result: dict = None, error: str = None):
    """Update job status"""
    if job_id not in jobs:
        return
    
    job = jobs[job_id]
    job.status = status
    if progress is not None:
        job.progress = progress
    if message is not None:
        job.message = message
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    
    if status == "running" and job.started_at is None:
        job.started_at = datetime.now().isoformat()
    elif status in ["completed", "failed"]:
        job.completed_at = datetime.now().isoformat()
    
    save_jobs(jobs)  # Persist to disk

def process_file_async(job_id: str, file_path: str, request: AnalysisRequest):
    """Process file in background thread"""
    try:
        update_job_status(job_id, "running", 10, "Starting pipeline...")
        
        # Create job-specific output directory
        output_dir = os.path.join("api_outputs", job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        update_job_status(job_id, "running", 20, "Running transcription...")
        
        # Run the pipeline
        result = run_pipeline(
            audio_file=file_path,
            output_dir=output_dir,
            formats=request.formats,
            confidence_threshold=request.confidence_threshold,
            language=request.language,
            run_llm_analysis=request.run_llm_analysis
        )
        
        if result["success"]:
            update_job_status(job_id, "completed", 100, "Analysis completed successfully", result)
        else:
            update_job_status(job_id, "failed", 0, "Pipeline failed", error=result.get("error"))
            
    except Exception as e:
        update_job_status(job_id, "failed", 0, "Unexpected error", error=str(e))
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

# --- API Endpoints ---

@app.get("/")
async def root():
    """API health check and info"""
    return {
        "message": "AI Conversation Analysis API",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "upload": "/analyze/",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}/{file_type}",
            "jobs": "/jobs/"
        }
    }

@app.post("/analyze/")
async def analyze_conversation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    confidence_threshold: float = Form(0.7),
    formats: str = Form("html,json,srt"),
    run_llm_analysis: bool = Form(True)
):
    """
    Upload and analyze an audio/video file
    
    - **file**: Audio or video file to analyze
    - **language**: Language code (auto, en, ar, etc.)
    - **confidence_threshold**: Minimum confidence for speaker assignment (0.0-1.0)
    - **formats**: Output formats (comma-separated: html,json,srt)
    - **run_llm_analysis**: Whether to run LLM meeting analysis
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {'.mp3', '.mp4', '.wav', '.m4a', '.avi', '.mov', '.mkv', '.webm', '.flac'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )
    
    # Create job
    job_id = create_job(file.filename)
    
    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Parse formats
    format_list = [f.strip() for f in formats.split(',')]
    
    # Create request object
    request = AnalysisRequest(
        language=language,
        confidence_threshold=confidence_threshold,
        formats=format_list,
        run_llm_analysis=run_llm_analysis
    )
    
    # Start background processing
    background_tasks.add_task(process_file_async, job_id, file_path, request)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "File uploaded successfully. Processing started.",
        "filename": file.filename
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return job.dict()

@app.get("/jobs/")
async def list_jobs():
    """List all jobs (last 50)"""
    job_list = list(jobs.values())
    # Sort by creation time, newest first
    job_list.sort(key=lambda x: x.created_at, reverse=True)
    return {"jobs": job_list[:50]}

@app.get("/download/{job_id}/{file_type}")
async def download_result(job_id: str, file_type: str):
    """
    Download result files
    
    - **job_id**: Job ID from the analyze endpoint
    - **file_type**: Type of file to download (html, json, srt, meeting_summary_txt, meeting_summary_html)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job.result or "outputs" not in job.result:
        raise HTTPException(status_code=404, detail="No outputs available")
    
    outputs = job.result["outputs"]
    if file_type not in outputs:
        available_types = list(outputs.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"File type '{file_type}' not available. Available types: {available_types}"
        )
    
    file_path = outputs[file_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine media type
    media_type_map = {
        "html": "text/html",
        "json": "application/json",
        "srt": "text/plain",
        "meeting_summary_txt": "text/plain",
        "meeting_summary_html": "text/html"
    }
    
    media_type = media_type_map.get(file_type, "application/octet-stream")
    filename = os.path.basename(file_path)
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its outputs"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Clean up output directory
    output_dir = os.path.join("api_outputs", job_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Remove from jobs dict
    del jobs[job_id]
    save_jobs(jobs)  # Persist to disk
    
    return {"message": "Job deleted successfully"}

@app.get("/stats/")
async def get_api_stats():
    """Get API usage statistics"""
    total_jobs = len(jobs)
    completed_jobs = sum(1 for job in jobs.values() if job.status == "completed")
    failed_jobs = sum(1 for job in jobs.values() if job.status == "failed")
    running_jobs = sum(1 for job in jobs.values() if job.status == "running")
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "running_jobs": running_jobs,
        "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    }

# --- Development Server ---
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Conversation Analysis API")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("🔧 Admin interface at: http://localhost:8000/redoc")
    
    # Create output directory
    os.makedirs("api_outputs", exist_ok=True)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,  # Use different port to avoid conflicts
        reload=False,  # Disable auto-reload to prevent job loss
        log_level="info"
    )
