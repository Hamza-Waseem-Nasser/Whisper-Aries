"""
conversation_renderer.py - Renders aligned conversation data to HTML, JSON, SRT, etc.

- Color-codes speakers
- Handles RTL for Arabic
- Includes metadata and confidence scores
- Exports to multiple formats
"""

from typing import List, Dict, Any
import json
import os

def render_html(aligned_segments: List[Dict], output_path: str, metadata: Dict = None):
    """
    Render aligned conversation to HTML with speaker colors and RTL support.
    """
    speaker_colors = {}
    color_palette = ["#4F8EF7", "#F78E4F", "#4FF78E", "#F74F8E", "#8E4FF7", "#F7F74F"]
    html = ["<html><head><meta charset='utf-8'><title>Conversation Transcript</title></head><body>"]
    html.append("<h2>Conversation Transcript</h2>")
    if metadata:
        html.append(f"<pre>{json.dumps(metadata, indent=2, ensure_ascii=False)}</pre>")
    for i, seg in enumerate(aligned_segments):
        speaker = seg.get('speaker', 'Unknown')
        if speaker not in speaker_colors:
            speaker_colors[speaker] = color_palette[len(speaker_colors) % len(color_palette)]
        color = speaker_colors[speaker]
        lang = seg.get('lang', 'auto')
        rtl = 'dir=\'rtl\'' if lang == 'ar' else ''
        conf = seg.get('confidence', 1.0)
        html.append(f"<div style='margin:8px 0;'><span style='color:{color};font-weight:bold;'>{speaker}</span> <span style='font-size:smaller;color:#888;'>[{seg['start']:.2f}-{seg['end']:.2f}s, conf={conf:.2f}]</span><br><span {rtl}>{seg['text']}</span></div>")
    html.append("</body></html>")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def render_json(aligned_segments: List[Dict], output_path: str, metadata: Dict = None):
    """
    Export aligned conversation to JSON.
    """
    out = {
        'segments': aligned_segments,
        'metadata': metadata or {}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def render_srt(aligned_segments: List[Dict], output_path: str):
    """
    Export aligned conversation to SRT format.
    """
    def srt_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, seg in enumerate(aligned_segments, 1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}")
        speaker = seg.get('speaker', 'Unknown')
        text = seg['text']
        lines.append(f"[{speaker}] {text}")
        lines.append("")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# --- Example usage (to be replaced with actual pipeline integration) ---
if __name__ == "__main__":
    aligned_segments = [
        {'text': 'Hello.', 'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00', 'confidence': 1.0},
        {'text': 'How are you?', 'start': 1.6, 'end': 3.0, 'speaker': 'SPEAKER_01', 'confidence': 1.0},
    ]
    metadata = {'test': True}
    render_html(aligned_segments, 'test_output.html', metadata)
    render_json(aligned_segments, 'test_output.json', metadata)
    render_srt(aligned_segments, 'test_output.srt')
    print("Sample outputs written.")
