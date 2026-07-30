# SPDX-License-Identifier: Apache-2.0
"""
Image processing utilities for VLM (Vision-Language Model) support.

This module provides functions for loading images from base64 data URIs,
extracting images from OpenAI-format messages, and computing image
hashes for prefix cache deduplication.
"""

import base64
import binascii
import hashlib
import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from ..exceptions import InvalidRequestError

_IMAGE_INPUT_ERROR = (
    "Image inputs must be base64 data URIs "
    "(data:image/...;base64,...). Remote URLs and local file paths are not supported."
)
_AUDIO_INPUT_ERROR = (
    "input_audio.data must be a base64 string or base64 data URI. "
    "Local file paths are not supported."
)
_VIDEO_INPUT_ERROR = (
    "Video inputs must be base64 data URIs "
    "(data:video/...;base64,...). Remote URLs and local file paths are not supported."
)
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_VIDEO_FRAMES = 128
VIDEO_SAMPLE_FPS = 2.0
MAX_VIDEO_WIDTH = 4096
MAX_VIDEO_HEIGHT = 2160
MAX_VIDEO_FPS = 120.0
MAX_VIDEO_DURATION_SECONDS = 300.0
MAX_VIDEO_SOURCE_FRAMES = 36_000
MAX_VIDEO_FRAME_PIXELS = 4096 * 2160
MAX_VIDEO_DECODED_BYTES = 256 * 1024 * 1024
MAX_VIDEO_DECODER_ALLOCATION_BYTES = 64 * 1024 * 1024
VIDEO_DECODE_TIMEOUT_SECONDS = 45.0
FFMPEG_BINARY = "/opt/homebrew/bin/ffmpeg"


@dataclass(frozen=True)
class VideoInput:
    """Decoded frames and their effective sampling rate."""

    frames: Any
    fps: float
    content_hash: str


def _decode_base64_data_uri(value: str, *, field: str) -> bytes:
    """Decode a base64 data URI, mapping malformed input to a request error."""
    if not isinstance(value, str):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    stripped = value.strip()
    if not stripped.startswith("data:"):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    prefix, separator, encoded = stripped.partition(",")
    prefix_lower = prefix.lower()
    if (
        separator != ","
        or not prefix_lower.startswith("data:image/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            f"{field} must use a base64 image data URI.",
            field=field,
        )

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{field} contains invalid base64 data.",
            field=field,
        ) from exc


def _decode_input_audio_data(data: str, *, field: str = "input_audio.data") -> bytes:
    """Decode input_audio.data without falling back to filesystem paths."""
    stripped = data.strip()
    if stripped.startswith("data:"):
        prefix, separator, encoded = stripped.partition(",")
        if separator != "," or ";base64" not in prefix.lower():
            raise InvalidRequestError(
                f"{field} must use a base64 data URI.",
                field=field,
            )
    else:
        encoded = stripped

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(_AUDIO_INPUT_ERROR, field=field) from exc


def _decode_video_data_uri(
    value: str,
    *,
    field: str = "video_url.url",
) -> tuple[bytes, str]:
    """Decode one inline video while rejecting paths, URLs, and oversized payloads."""
    if not isinstance(value, str):
        raise InvalidRequestError(_VIDEO_INPUT_ERROR, field=field)

    stripped = value.strip()
    if not stripped.startswith("data:"):
        raise InvalidRequestError(_VIDEO_INPUT_ERROR, field=field)

    prefix, separator, encoded = stripped.partition(",")
    prefix_lower = prefix.lower()
    if (
        separator != ","
        or not prefix_lower.startswith("data:video/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            f"{field} must use a base64 video data URI.",
            field=field,
        )

    max_encoded_size = ((MAX_VIDEO_BYTES + 2) // 3) * 4
    if len(encoded) > max_encoded_size:
        raise InvalidRequestError(
            f"{field} exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)} MiB limit.",
            field=field,
        )

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{field} contains invalid base64 data.",
            field=field,
        ) from exc
    if not raw:
        raise InvalidRequestError(f"{field} is empty.", field=field)
    if len(raw) > MAX_VIDEO_BYTES:
        raise InvalidRequestError(
            f"{field} exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)} MiB limit.",
            field=field,
        )

    media_type = prefix_lower[5:].split(";", 1)[0]
    suffix = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/mpeg": ".mpeg",
        "video/x-msvideo": ".avi",
    }.get(media_type, ".video")
    return raw, suffix


def _video_limit_error(field: str) -> InvalidRequestError:
    return InvalidRequestError(
        f"{field} exceeds safe decoded video limits.",
        field=field,
    )


def _validated_video_sample_count(
    *,
    width: float,
    height: float,
    fps: float,
    total_frames: float,
    field: str,
) -> int:
    """Validate decoder metadata and return a bounded even sample count."""
    values = (width, height, fps, total_frames)
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    ):
        raise _video_limit_error(field)

    int_width = int(width)
    int_height = int(height)
    int_total_frames = int(total_frames)
    if (
        int_width != width
        or int_height != height
        or int_total_frames != total_frames
        or int_width <= 0
        or int_height <= 0
        or int_total_frames < 2
        or fps <= 0
        or int_width > MAX_VIDEO_WIDTH
        or int_height > MAX_VIDEO_HEIGHT
        or fps > MAX_VIDEO_FPS
        or int_total_frames > MAX_VIDEO_SOURCE_FRAMES
        or int_width * int_height > MAX_VIDEO_FRAME_PIXELS
        or int_total_frames / fps > MAX_VIDEO_DURATION_SECONDS
    ):
        raise _video_limit_error(field)

    frame_factor = 2
    minimum_frames = min(4, int_total_frames)
    maximum_frames = min(MAX_VIDEO_FRAMES, int_total_frames)
    maximum_frames -= maximum_frames % frame_factor
    desired = int_total_frames / fps * VIDEO_SAMPLE_FPS
    sample_count = min(max(desired, minimum_frames), maximum_frames)
    sample_count = math.floor(sample_count / frame_factor) * frame_factor
    if sample_count < frame_factor or sample_count > int_total_frames:
        raise _video_limit_error(field)

    # Reserve one decoder frame and one RGB conversion frame in addition to
    # the retained output tensor so peak decoded memory stays within the cap.
    decoded_bytes = (sample_count + 2) * int_width * int_height * 3
    if decoded_bytes > MAX_VIDEO_DECODED_BYTES:
        raise _video_limit_error(field)
    return sample_count


def _decode_video_file_to_memmap(
    path: str,
    output_path: str,
    *,
    field: str,
) -> tuple[tuple[int, int, int, int], float]:
    """Decode bounded RGB frames with ffmpeg allocation controls."""
    import subprocess

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise InvalidRequestError(
                f"{field} does not contain a decodable video.", field=field
            )

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        sample_count = _validated_video_sample_count(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            field=field,
        )
        int_width = int(width)
        int_height = int(height)
        int_total_frames = int(total_frames)
        indices = np.linspace(0, int_total_frames - 1, sample_count).round().astype(int)

        select = "+".join(f"eq(n\\,{int(index)})" for index in indices)
        video_filter = (
            f"select={select},"
            f"scale={int_width}:{int_height}:flags=area,format=rgb24"
        )
        command = [
            FFMPEG_BINARY,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-max_alloc",
            str(MAX_VIDEO_DECODER_ALLOCATION_BYTES),
            "-threads",
            "1",
            "-i",
            path,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            video_filter,
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(sample_count),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-y",
            output_path,
        ]
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={
                "HOME": "/var/empty",
                "LANG": "C",
                "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
            },
            timeout=VIDEO_DECODE_TIMEOUT_SECONDS,
            check=False,
        )
        output_shape = (sample_count, int_height, int_width, 3)
        expected_size = math.prod(output_shape)
        if result.returncode != 0 or os.path.getsize(output_path) != expected_size:
            raise InvalidRequestError(
                f"{field} does not contain a decodable video.", field=field
            )
        sample_fps = sample_count / int_total_frames * fps
        return output_shape, float(sample_fps)
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise InvalidRequestError(
            f"{field} does not contain a decodable video.", field=field
        ) from exc
    finally:
        cap.release()


def _load_video_file_bounded(path: str, *, field: str) -> tuple[Any, float]:
    """Decode video into one bounded memmap and return a zero-copy CHW view."""
    import numpy as np

    output_file = tempfile.NamedTemporaryFile(suffix=".video-rgb", delete=False)
    output_path = output_file.name
    output_file.close()
    try:
        shape, fps = _decode_video_file_to_memmap(
            path, output_path, field=field
        )
        frames_hwc = np.memmap(
            output_path,
            mode="r",
            dtype=np.uint8,
            shape=tuple(shape),
        )
        os.unlink(output_path)
        return np.transpose(frames_hwc, (0, 3, 1, 2)), float(fps)
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise InvalidRequestError(
            f"{field} does not contain a decodable video.", field=field
        ) from exc
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def load_video_data_uri(value: str, *, field: str = "video_url.url") -> VideoInput:
    """Sample a bounded inline video without exposing filesystem reads."""
    raw, suffix = _decode_video_data_uri(value, field=field)
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(raw)
        tmp.flush()
        frames, fps = _load_video_file_bounded(tmp.name, field=field)
    return VideoInput(
        frames=frames,
        fps=fps,
        content_hash=hashlib.sha256(raw).hexdigest(),
    )


def validate_image_data_uri(value: str, *, field: str = "image") -> str:
    """Validate that a request-facing image reference is an inline data URI."""
    _decode_base64_data_uri(value, field=field)
    return value


def load_image(url_or_base64: str, *, field: str = "image_url") -> Image.Image:
    """
    Load an image from a base64 data URI.

    Supports:
    - Data URIs: "data:image/jpeg;base64,..." format

    Args:
        url_or_base64: Image base64 data URI string

    Returns:
        PIL Image object

    Raises:
        InvalidRequestError: If the input is not a valid image data URI
    """
    img_bytes = _decode_base64_data_uri(url_or_base64, field=field)
    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        raise InvalidRequestError(
            f"{field} does not contain a decodable image.",
            field=field,
        ) from exc

    # Apply EXIF orientation (phone photos etc.) before processing.
    # Matches mlx-vlm's load_image which calls ImageOps.exif_transpose().
    img = ImageOps.exif_transpose(img)
    # Ensure RGB format (RGBA/P/L etc. cause broadcast errors in vision processors)
    return img.convert("RGB")


def _extract_media_from_messages(
    messages: List[Dict[str, Any]],
    *,
    include_videos: bool,
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List, List[VideoInput]]:
    """
    Extract images, audio, and optionally video from OpenAI-format messages.

    Processes messages containing content arrays with image_url, input_audio,
    or video_url parts, loads the media, and returns cleaned text-only messages
    alongside the loaded media.

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts
            (text/image_url/input_audio/video_url).
        include_videos: Whether to decode and return video_url parts.

    Returns:
        Tuple of (text_messages, images, audio, videos):
        - text_messages: Messages with media parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
        - audio: List of BytesIO audio buffers
        - videos: List containing at most one sampled VideoInput
    """
    text_messages = []
    images = []
    audio = []
    videos: list[VideoInput] = []

    if include_videos:
        video_parts = 0
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = (
                    part.get("type")
                    if isinstance(part, dict)
                    else getattr(part, "type", None)
                )
                if part_type in ("video_url", "input_video"):
                    if message.get("role", "user") != "user":
                        raise InvalidRequestError(
                            "Video input is only supported in user messages.",
                            field="messages",
                        )
                    video_parts += 1
        if video_parts > 1:
            raise InvalidRequestError(
                "Only one video is supported per request.",
                field="messages",
            )

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            # Simple string content — pass through
            text_messages.append({"role": role, "content": content or ""})
            # Preserve extra fields (tool_calls, tool_call_id, etc.)
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        # Content array with text, image_url, and/or input_audio parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                # Pydantic model (ContentPart)
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                # OpenAI chat format: {"type":"image_url","image_url":{"url":"..."}}
                # Responses-style format: {"type":"input_image","image_url":"..."}
                image_url_obj = (
                    part.get("image_url")
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url_obj is None and isinstance(part, dict):
                    image_url_obj = part.get("input_image")

                url = None
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                elif isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url")
                elif image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)

                if url:
                    images.append(load_image(url, field="image_url"))

            elif part_type == "input_audio":
                # OpenAI audio format: {"type":"input_audio","input_audio":{"data":"...","format":"wav"}}
                input_audio = (
                    part.get("input_audio")
                    if isinstance(part, dict)
                    else getattr(part, "input_audio", None)
                )
                if input_audio and isinstance(input_audio, dict):
                    data = input_audio.get("data", "")
                    if isinstance(data, str):
                        audio.append(io.BytesIO(_decode_input_audio_data(data)))
                    elif isinstance(data, bytes):
                        audio.append(io.BytesIO(data))
                    else:
                        audio.append(data)

            elif part_type in ("video_url", "input_video") and include_videos:
                video_url_obj = (
                    part.get("video_url")
                    if isinstance(part, dict)
                    else getattr(part, "video_url", None)
                )
                if video_url_obj is None and isinstance(part, dict):
                    video_url_obj = part.get("input_video")

                if isinstance(video_url_obj, str):
                    url = video_url_obj
                elif isinstance(video_url_obj, dict):
                    url = video_url_obj.get("url")
                elif video_url_obj is not None:
                    url = getattr(video_url_obj, "url", None)
                else:
                    url = None

                if not url:
                    raise InvalidRequestError(
                        "video_url.url is required.",
                        field="video_url.url",
                    )
                if videos:
                    raise InvalidRequestError(
                        "Only one video is supported per request.",
                        field="messages",
                    )
                videos.append(load_video_data_uri(url))

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images, audio, videos


def extract_media_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List, List[VideoInput]]:
    """Extract all request media, including one inline video."""
    return _extract_media_from_messages(messages, include_videos=True)


def extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List]:
    """Backward-compatible image/audio extractor that ignores video parts."""
    text_messages, images, audio, _ = _extract_media_from_messages(
        messages,
        include_videos=False,
    )
    return text_messages, images, audio


def compute_image_hash(images: List[Image.Image]) -> Optional[str]:
    """
    Compute a SHA256 hash from a list of images for prefix cache deduplication.

    Uses image size and raw pixel data to produce a deterministic hash.
    Returns None if images list is empty.

    Args:
        images: List of PIL Image objects

    Returns:
        Hex-encoded SHA256 hash string, or None if no images
    """
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        # Include image dimensions
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        # Include raw pixel data (convert to RGB for consistency)
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())

    return hasher.hexdigest()


def compute_per_image_hashes(images: List[Image.Image]) -> List[str]:
    """Compute individual SHA256 hashes for each image.

    Returns a list of hex-encoded hash strings, one per image.
    """
    hashes = []
    for img in images:
        hasher = hashlib.sha256()
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())
        hashes.append(hasher.hexdigest())
    return hashes
