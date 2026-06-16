from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..errors import PreprocessViolation



def probe_video_meta(video_path: str, ffprobe_path: str, source_platform: str) -> dict:
    args = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreprocessViolation(f"ffprobe 失败: {proc.stderr[:300]}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise PreprocessViolation("视频缺少 video stream")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    rate = str(stream.get("r_frame_rate") or "0/1")
    num, den = rate.split("/")
    fps = round(float(num) / float(den), 3) if float(den) else 0.0
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    if duration <= 0:
        raise PreprocessViolation("视频时长非法")
    return {
        "source_platform": source_platform or "unknown",
        "duration_sec": round(duration, 3),
        "fps": fps,
        "resolution": f"{width}x{height}",
    }



def extract_audio(video_path: str, audio_path: str, ffmpeg_path: str) -> None:
    args = [ffmpeg_path, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_path]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreprocessViolation(f"音频抽取失败: {proc.stderr[:300]}")



def extract_frame(video_path: str, image_path: str, second: float, ffmpeg_path: str) -> None:
    args = [ffmpeg_path, "-y", "-ss", str(second), "-i", video_path, "-frames:v", "1", image_path]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PreprocessViolation(f"关键帧抽取失败: {proc.stderr[:300]}")
