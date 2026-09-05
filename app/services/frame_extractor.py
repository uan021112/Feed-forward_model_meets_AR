"""Video frame extraction service.

Extracts frames at a configurable rate (env FRAME_EXTRACT_FPS, default 1 fps)
after skipping any initial black-screen segment. Each frame is resized to the
largest resolution with total pixels ≤ 250,000 and both dimensions divisible
by 14, then saved as JPEG (quality=85) with absolute timestamp filenames.

Decoding uses FFmpeg subprocess piping for robust video I/O, replacing
OpenCV's VideoCapture which fails to seek in certain H.264/H.265 streams.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.config import get_frame_extract_fps
from app.utils.image import compute_resize_resolution


logger = logging.getLogger(__name__)

FRAME_INTERVAL_MS = 1000.0 / get_frame_extract_fps()
MAX_PIXELS = 250_000
ALIGN_DIVISOR = 14
JPEG_QUALITY = 85
BLACK_FRAME_MEAN_THRESHOLD = 10.0  # mean pixel intensity below this is black
MAX_BLACK_FRAME_SKIP_RATIO = 0.5  # if >50% of frames skipped, give up scanning


def _is_black_frame(frame: np.ndarray) -> bool:
    """Return True if the frame's mean pixel intensity is below the black threshold."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < BLACK_FRAME_MEAN_THRESHOLD
    except Exception:
        logger.warning("Black-frame check failed on a frame, treating as non-black", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# FFmpeg / ffprobe helpers
# ---------------------------------------------------------------------------

def _get_video_info(video_path: Path) -> dict:
    """Return video metadata via ffprobe.

    Returns:
        dict with keys: ``width``, ``height``, ``fps``, ``duration_ms``,
        ``total_frames``.
    """
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found — install FFmpeg: apt install ffmpeg")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"ffprobe failed on {video_path}: {exc.stderr.strip()}") from exc

    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    width: int = video_stream["width"]
    height: int = video_stream["height"]

    # Prefer avg_frame_rate — r_frame_rate can be the container timebase
    # (e.g. "90000/1") on some Android-encoded videos.
    fps_str: str = video_stream.get("avg_frame_rate", "0/1")
    if fps_str in ("0/1", "0/0"):
        fps_str = video_stream.get("r_frame_rate", "0/1")
    num_s, den_s = fps_str.split("/")
    num, den = float(num_s), float(den_s)
    fps = num / den if den != 0 else 0.0
    if fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps_str}")

    duration_s = float(data.get("format", {}).get("duration", 0))
    duration_ms = duration_s * 1000.0

    nb_frames_raw = video_stream.get("nb_frames")
    if nb_frames_raw is not None and int(nb_frames_raw) > 0:
        total_frames = int(nb_frames_raw)
    else:
        total_frames = max(1, int(duration_s * fps))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_ms": duration_ms,
        "total_frames": total_frames,
    }


def _decode_video_frames(
    video_path: Path,
    width: int,
    height: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame_ndarray)`` by piping raw BGR24 from ffmpeg.

    The generator reads frames sequentially — only one frame is materialised
    at a time, so memory usage stays constant regardless of video length.
    """
    frame_size = width * height * 3  # BGR24: 3 bytes/pixel
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found — install FFmpeg: apt install ffmpeg")

    assert proc.stdout is not None
    assert proc.stderr is not None
    frame_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            yield frame_idx, frame
            frame_idx += 1
    finally:
        proc.stdout.close()
        # Drain stderr to avoid deadlock, then wait
        _ = proc.stderr.read()
        returncode = proc.wait()
        # Exit code 1 is normal when the pipe is closed early (SIGPIPE).
        # Only warn when *no* frames were decoded — that signals a real failure.
        if returncode != 0 and frame_idx == 0:
            logger.warning(
                "ffmpeg exited with code %d for %s", returncode, video_path,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    output_dir: Path,
    video_start_timestamp: int,
) -> list[tuple[int, Path]]:
    """Extract frames from video at the configured fps after skipping black-screen frames.

    Scans forward past frames whose mean pixel intensity falls below
    ``BLACK_FRAME_MEAN_THRESHOLD``, then extracts frames at the rate
    configured by env ``FRAME_EXTRACT_FPS`` (default 1 fps). Each frame is
    resized to the largest resolution with total pixels ≤ 250,000 and both
    dimensions divisible by 14, then saved as JPEG (quality=85) with its
    absolute timestamp as the filename.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to save extracted JPEG frames.
        video_start_timestamp: Absolute epoch timestamp (ms) of the video's
            first frame, provided by the frontend.

    Returns:
        List of (absolute_timestamp_ms, saved_filepath) tuples.

    Raises:
        ValueError: If the video cannot be opened, has invalid properties,
            or no valid frames could be extracted.
    """
    info = _get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]
    duration_ms = info["duration_ms"]
    width = info["width"]
    height = info["height"]

    target_w, target_h = compute_resize_resolution(
        width, height,
        max_pixels=MAX_PIXELS,
        divisor=ALIGN_DIVISOR,
    )

    extract_fps = 1000.0 / FRAME_INTERVAL_MS
    logger.info(
        "Starting frame extraction: %d total frames at %.2f native fps, "
        "extracting at %.1f fps, black-frame threshold=%.0f, max skip ratio=%.0f%%",
        total_frames, fps, extract_fps, BLACK_FRAME_MEAN_THRESHOLD,
        MAX_BLACK_FRAME_SKIP_RATIO * 100,
    )

    extracted: list[tuple[int, Path]] = []
    scanning_black = True
    black_skip_count = 0
    next_extract_ms = 0.0

    for frame_idx, frame in _decode_video_frames(video_path, width, height):
        frame_time_ms = (frame_idx / fps) * 1000.0

        # Skip initial black-screen segment, with a sanity cap
        if scanning_black:
            if _is_black_frame(frame):
                black_skip_count += 1
                if black_skip_count > int(total_frames * MAX_BLACK_FRAME_SKIP_RATIO):
                    logger.warning(
                        "Skipped %d black frames (>%.0f%% of %d total) — "
                        "likely dim scene, stopping black-frame scan at frame %d (%.1fs)",
                        black_skip_count, MAX_BLACK_FRAME_SKIP_RATIO * 100,
                        total_frames, frame_idx, frame_time_ms / 1000.0,
                    )
                    scanning_black = False
                    next_extract_ms = frame_time_ms
                else:
                    continue
            else:
                logger.info(
                    "Black-screen scan complete: skipped %d black frames, "
                    "first content frame at #%d (%.1fs)",
                    black_skip_count, frame_idx, frame_time_ms / 1000.0,
                )
                scanning_black = False
                next_extract_ms = frame_time_ms

        if frame_time_ms > duration_ms:
            break

        # Extract when we reach the next extraction boundary
        if frame_time_ms >= next_extract_ms:
            abs_timestamp = video_start_timestamp + int(frame_time_ms)

            resized = cv2.resize(frame, (target_w, target_h))
            out_path = output_dir / f"{abs_timestamp}.jpg"
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            extracted.append((abs_timestamp, out_path))
            next_extract_ms += FRAME_INTERVAL_MS

    if not extracted:
        logger.error(
            "Frame extraction failed: 0 frames extracted from %d total "
            "(%d black frames skipped, video duration %.1fs)",
            total_frames, black_skip_count, duration_ms / 1000.0,
        )
        raise ValueError("No frames could be extracted from the video")

    logger.info(
        "Frame extraction complete: %d frames extracted, %d black frames skipped "
        "(%d total frames, %.1fs video)",
        len(extracted), black_skip_count, total_frames, duration_ms / 1000.0,
    )
    return extracted


def extract_frames_at_timestamps(
    video_path: Path,
    output_dir: Path,
    video_start_timestamp: int,
    target_timestamps_ms: list[int],
) -> list[tuple[int, Path]]:
    """Extract frames at exact timestamps from a video.

    Decodes the video in a single FFmpeg pass and matches each requested
    timestamp to its **temporally closest** decoded frame. Black frames
    are skipped.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to save extracted JPEG frames.
        video_start_timestamp: Absolute epoch timestamp (ms) of the video's
            first frame.
        target_timestamps_ms: Absolute timestamps (ms) for frame extraction,
            sorted ascending.

    Returns:
        List of ``(timestamp_ms, saved_filepath)`` tuples.

    Raises:
        ValueError: If the video cannot be opened or no valid frames could
            be extracted.
    """
    if not target_timestamps_ms:
        raise ValueError("No target timestamps provided")

    info = _get_video_info(video_path)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]
    frame_ms_per_frame = 1000.0 / fps

    target_w, target_h = compute_resize_resolution(
        width, height,
        max_pixels=MAX_PIXELS,
        divisor=ALIGN_DIVISOR,
    )

    # Build (absolute_ts, rel_ms_in_video) pairs, clamped to ≥ 0
    sorted_ts = sorted(target_timestamps_ms)
    targets: list[tuple[int, float]] = []
    for ts in sorted_ts:
        rel_ms = ts - video_start_timestamp
        if rel_ms < 0:
            logger.warning("Timestamp %d is before video start; clamping to 0", ts)
            rel_ms = 0.0
        targets.append((ts, float(rel_ms)))

    extracted: list[tuple[int, Path]] = []
    t_idx = 0
    n_targets = len(targets)

    prev_frame: np.ndarray | None = None
    prev_idx: int = -1

    for frame_idx, frame in _decode_video_frames(video_path, width, height):
        frame_time_ms = frame_idx * frame_ms_per_frame

        # Commit every target we have just passed (target_rel ≤ frame_time)
        while t_idx < n_targets and targets[t_idx][1] <= frame_time_ms:
            ts, target_rel = targets[t_idx]

            # Pick the temporally closer of (prev_frame, current_frame)
            if prev_frame is not None:
                prev_time = prev_idx * frame_ms_per_frame
                if abs(prev_time - target_rel) <= abs(frame_time_ms - target_rel):
                    best_frame = prev_frame
                else:
                    best_frame = frame
            else:
                best_frame = frame

            if _is_black_frame(best_frame):
                logger.info("Skipping black frame at timestamp %d", ts)
            else:
                resized = cv2.resize(best_frame, (target_w, target_h))
                out_path = output_dir / f"{ts}.jpg"
                cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                extracted.append((ts, out_path))

            t_idx += 1

        if t_idx >= n_targets:
            break  # all targets matched; stop decoding early

        prev_frame = frame
        prev_idx = frame_idx

    # Any targets left over are beyond the end of the video
    for idx in range(t_idx, n_targets):
        ts, target_rel = targets[idx]
        logger.warning(
            "Timestamp %d (%.0f ms into video) is beyond video end; no frame extracted",
            ts, target_rel,
        )

    if not extracted:
        raise ValueError(
            "No frames could be extracted from the video at the requested timestamps",
        )

    logger.info(
        "Timestamp-based frame extraction complete: %d frames extracted "
        "from %d requested timestamps",
        len(extracted), n_targets,
    )
    return extracted
