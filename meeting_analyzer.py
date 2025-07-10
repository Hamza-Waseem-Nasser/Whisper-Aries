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

# Initialize OpenAI client for v1.x API
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an AI meeting assistant. Given the following transcript (with speaker labels and timestamps), please:
1. Summarize the main points and decisions.
2. List action items.
3. For each speaker, report how much they spoke (in seconds and as a percentage).
4. Highlight any important questions or issues raised.

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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI meeting assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
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
