"""Bounded media downloads and full-frame video extraction for the demo."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from .video_processor import process_video


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/quicktime"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
DEFAULT_VIDEO_MAX_BYTES = 16 * 1024 * 1024


class InvalidVideo(ValueError):
    """The received video cannot yield a usable frame."""


def read_media_response(response, max_bytes: int) -> bytes:
    length = response.headers.get("content-length")
    if length:
        try:
            declared_length = int(length)
        except ValueError as exc:
            raise ValueError("Invalid media length") from exc
        if declared_length > max_bytes:
            raise ValueError("Media exceeds the demo size limit")

    chunks = []
    size = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > max_bytes:
            raise ValueError("Media exceeds the demo size limit")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("Media is empty")
    return b"".join(chunks)


def extract_best_frame(video_bytes: bytes, content_type: str) -> bytes:
    if content_type not in SUPPORTED_VIDEO_TYPES:
        raise InvalidVideo("Unsupported video format")
    suffix = ".mov" if content_type == "video/quicktime" else ".mp4"
    with TemporaryDirectory(prefix="canine-video-") as directory:
        video_path = Path(directory) / f"incoming{suffix}"
        frame_path = Path(directory) / "best_frame.jpg"
        video_path.write_bytes(video_bytes)
        try:
            result = process_video(str(video_path), output_path=str(frame_path), top_k=1)
        except (OSError, ValueError, cv2.error) as exc:
            raise InvalidVideo("Video could not be decoded") from exc
        if result.get("best_frame") != str(frame_path) or not frame_path.is_file():
            raise InvalidVideo("No frame could be extracted")
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise InvalidVideo("Extracted frame is unreadable")
        return frame_path.read_bytes()
