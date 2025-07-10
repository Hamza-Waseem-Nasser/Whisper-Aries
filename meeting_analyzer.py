"""
meeting_analyzer.py - LLM-powered meeting summarization and analytics

- Loads conversation JSON output (speaker, text, timestamps)
- Sends to Gemini LLM API for summary, action items, and speaker stats
- Saves results as text and HTML
"""

import json

import openai
from pathlib import Path
from collections import defaultdict
import os

# Initialize OpenAI client for v1.x API - handle missing API key gracefully
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not set. LLM analysis will be skipped.")
        client = None
    else:
        client = openai.OpenAI(api_key=api_key)
except Exception as e:
    print(f"⚠️ OpenAI client initialization failed: {e}")
    client = None

PROMPT_TEMPLATE = """
You are an expert AI meeting and interview assistant. Given the following transcript (with speaker labels and timestamps), your tasks are:
1. Write a detailed summary of the conversation, focusing on key points, decisions, and any important context.
2. Identify and list all action items, tasks, or follow-ups discussed.
3. For each speaker, report how much they spoke (in seconds and as a percentage), and summarize their main contributions or questions.
4. Highlight any important questions, issues, or topics raised.
5. If this is an interview, extract candidate strengths, weaknesses, and any hiring recommendations.
6. If this is a business meeting, extract decisions, blockers, and next steps.
7. Make your output as detailed and structured as possible, using bullet points and sections where appropriate.

Transcript:
{transcript}
"""

def load_transcript(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['segments'] if 'segments' in data else data

def build_transcript_text(segments):
    lines = []
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '')
        lines.append(f"[{speaker}] {start:.2f}-{end:.2f}: {text}")
    return '\n'.join(lines)

def speaker_stats(segments):
    stats = defaultdict(float)
    total = 0.0
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        duration = seg.get('end', 0) - seg.get('start', 0)
        stats[speaker] += duration
        total += duration
    return {s: {'seconds': t, 'percent': (t/total*100 if total else 0)} for s, t in stats.items()}


# --- OpenAI LLM Call ---

# Updated for openai>=1.0.0
def call_openai(prompt, model="gpt-4o"):
    if not client:
        raise Exception("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert AI meeting and interview assistant. Always provide detailed, structured, and actionable insights for interviews, business meetings, and conversations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=8192,  # Increased for more detailed output
        temperature=0.3,
    )
    return response.choices[0].message.content

def save_results(base_path, summary, stats):
    txt_path = base_path.with_suffix('.meeting_summary.txt')
    html_path = base_path.with_suffix('.meeting_summary.html')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary)
        f.write('\n\nSpeaker Stats:\n')
        for s, v in stats.items():
            f.write(f"{s}: {v['seconds']:.1f}s ({v['percent']:.1f}%)\n")
    html = ["<html><head><meta charset='utf-8'><title>Meeting Summary</title></head><body>"]
    html.append("<h2>Meeting Summary</h2>")
    html.append(f"<pre>{summary}</pre>")
    html.append("<h3>Speaker Stats</h3><ul>")
    for s, v in stats.items():
        html.append(f"<li><b>{s}</b>: {v['seconds']:.1f}s ({v['percent']:.1f}%)</li>")
    html.append("</ul></body></html>")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"✅ Meeting summary saved: {txt_path}\n✅ HTML summary saved: {html_path}")


def analyze_meeting(json_path):
    segments = load_transcript(json_path)
    transcript = build_transcript_text(segments)
    stats = speaker_stats(segments)
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)
    print("🧠 Sending transcript to OpenAI GPT for analysis...")
    summary = call_openai(prompt)
    save_results(Path(json_path), summary, stats)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python meeting_analyzer.py <transcript.json>")
        sys.exit(1)
    analyze_meeting(sys.argv[1])
