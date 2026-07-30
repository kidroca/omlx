# SPDX-License-Identifier: Apache-2.0
"""Tests for utils/image.py — image loading, extraction, and hashing."""

import base64
import hashlib
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import omlx.utils.image as image_utils
from omlx.exceptions import InvalidRequestError
from omlx.utils.image import (
    compute_image_hash,
    compute_per_image_hashes,
    extract_images_from_messages,
    load_image,
)

# =============================================================================
# Helper: create small test images
# =============================================================================


def _make_test_image(
    width: int = 4, height: int = 4, color: str = "red"
) -> Image.Image:
    """Create a small solid-color test image."""
    return Image.new("RGB", (width, height), color)


def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# Tests: load_image
# =============================================================================


class TestLoadImage:
    """Tests for load_image()."""

    def test_load_base64_png(self):
        """Load image from data URI with base64 PNG."""
        img = _make_test_image(8, 8, "blue")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (8, 8)

    def test_rgba_converted_to_rgb(self):
        """RGBA images (e.g. transparent PNGs) are converted to RGB."""
        rgba_img = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        b64 = _image_to_base64(rgba_img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.mode == "RGB"

    def test_load_base64_jpeg(self):
        """Load image from data URI with base64 JPEG."""
        img = _make_test_image(8, 8, "green")
        b64 = _image_to_base64(img, "JPEG")
        uri = f"data:image/jpeg;base64,{b64}"

        loaded = load_image(uri)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (8, 8)

    def test_rejects_data_uri_without_image_media_type(self):
        """Image data URIs must include an image media type."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        uri = f"data:;base64,{b64}"

        with pytest.raises(InvalidRequestError):
            load_image(uri)

    @patch("urllib.request.urlopen")
    def test_rejects_remote_url_without_fetching(self, mock_urlopen):
        """Remote URL image refs are rejected without server-side fetches."""
        with pytest.raises(InvalidRequestError):
            load_image("https://example.com/image.png")
        mock_urlopen.assert_not_called()

    def test_rejects_local_file_path(self, tmp_path):
        """Local filesystem image refs are rejected before opening files."""
        img = _make_test_image(4, 4)
        path = tmp_path / "local.png"
        img.save(path)

        with pytest.raises(InvalidRequestError):
            load_image(str(path))

    def test_load_invalid_format_raises(self):
        """Invalid input raises a request error."""
        with pytest.raises(InvalidRequestError):
            load_image("not-a-valid-image-source")

    def test_load_invalid_base64_raises(self):
        """Invalid base64 data raises error."""
        with pytest.raises(InvalidRequestError):
            load_image("data:image/png;base64,not_valid_base64!!!")

    def test_rejects_non_image_data_uri(self):
        """Non-image data URIs are rejected for image inputs."""
        data = base64.b64encode(b"hello").decode()
        with pytest.raises(InvalidRequestError):
            load_image(f"data:text/plain;base64,{data}")


# =============================================================================
# Tests: extract_images_from_messages
# =============================================================================


class TestExtractImagesFromMessages:
    """Tests for extract_images_from_messages()."""

    def test_text_only_messages(self):
        """Text-only messages return empty image list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(text_msgs) == 2
        assert len(images) == 0
        assert len(audio) == 0
        assert text_msgs[0]["content"] == "Hello"

    def test_message_with_image_url(self):
        """Messages with image_url content parts extract images."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": "What is this?"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        # Text-only message should contain only text part
        assert text_msgs[0]["role"] == "user"

    def test_message_with_input_image(self):
        """Messages with input_image content parts extract images."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": uri},
                    {"type": "input_text", "text": "What is this?"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert text_msgs[0]["role"] == "user"

    def test_multiple_images_in_one_message(self):
        """Multiple images in a single message are all extracted."""
        img1 = _make_test_image(4, 4, "red")
        img2 = _make_test_image(4, 4, "blue")
        b64_1 = _image_to_base64(img1)
        b64_2 = _image_to_base64(img2)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_1}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_2}"},
                    },
                    {"type": "text", "text": "Compare these"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 2

    def test_mixed_text_and_image_messages(self):
        """Mix of text-only and image messages."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": "Describe this"},
                ],
            },
            {"role": "assistant", "content": "I see an image."},
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1
        assert len(text_msgs) == 3
        # System message preserved as-is
        assert text_msgs[0]["content"] == "You are helpful."

    def test_preserves_extra_fields(self):
        """Extra fields like tool_calls are preserved."""
        messages = [
            {
                "role": "assistant",
                "content": "Using tool",
                "tool_calls": [{"id": "tc1", "function": {"name": "test"}}],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert "tool_calls" in text_msgs[0]

    def test_invalid_image_url_rejected(self):
        """Invalid image URLs are rejected instead of silently skipped."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "invalid://not-real"}},
                    {"type": "text", "text": "test"},
                ],
            },
        ]
        with pytest.raises(InvalidRequestError):
            extract_images_from_messages(messages)

    def test_pydantic_model_content_parts(self):
        """Content parts as Pydantic-like objects with type/text/image_url attrs."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)

        # Simulate Pydantic ContentPart with image_url
        image_part = MagicMock(spec=[])
        image_part.type = "image_url"
        image_url = MagicMock(spec=[])
        image_url.url = f"data:image/png;base64,{b64}"
        image_part.image_url = image_url

        # Simulate Pydantic ContentPart with text
        text_part = MagicMock(spec=[])
        text_part.type = "text"
        text_part.text = "What?"

        messages = [
            {"role": "user", "content": [image_part, text_part]},
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1

    def test_input_audio_base64_data_uri(self):
        """Messages with input_audio base64 data URI extract audio."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAABCxAgAEABAAZGF0YQAAAAA=",
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "What do you hear?"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(audio) == 1
        assert len(images) == 0
        # Audio should be a BytesIO object
        assert hasattr(audio[0], "read")

    def test_input_audio_raw_base64(self):
        """Messages with raw base64 input_audio extract audio."""
        import base64

        raw_bytes = b"\x00\x01\x02\x03" * 16
        b64 = base64.b64encode(raw_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(audio) == 1
        assert hasattr(audio[0], "read")

    def test_input_audio_bytes_data(self):
        """Messages with bytes input_audio.data extract audio."""
        raw_bytes = b"\x00\x01\x02\x03" * 16

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": raw_bytes,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(audio) == 1
        assert hasattr(audio[0], "read")

    def test_input_audio_string_path_rejected(self):
        """Non-base64 input_audio.data strings are rejected as path refs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "/tmp/audio.wav",
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        with pytest.raises(InvalidRequestError):
            extract_images_from_messages(messages)

    def test_invalid_input_audio_rejected(self):
        """Invalid input_audio base64 is rejected."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64,!!!not_valid_base64!!!",
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "test"},
                ],
            },
        ]
        with pytest.raises(InvalidRequestError):
            extract_images_from_messages(messages)

    def test_audio_mixed_with_images(self):
        """Audio and images in the same message both extracted."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        import base64 as b64_mod

        raw_bytes = b"\x00\x01\x02\x03" * 16
        audio_b64 = b64_mod.b64encode(raw_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "Describe this image and audio"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1
        assert len(audio) == 1
        # Text content should be preserved
        assert "Describe this image and audio" in text_msgs[0]["content"]


class TestExtractMediaFromMessages:
    """Tests for native inline-video extraction."""

    def test_video_url_is_decoded_loaded_and_removed_from_text(self):
        raw_video = b"small synthetic mp4 payload"
        uri = "data:video/mp4;base64," + base64.b64encode(raw_video).decode()
        observed = {}

        def fake_load_video(path, *, field):
            observed["path"] = path
            with open(path, "rb") as video_file:
                observed["bytes"] = video_file.read()
            observed["field"] = field
            return "frames", 1.75

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                    {"type": "text", "text": "Describe this video"},
                ],
            }
        ]

        with patch(
            "omlx.utils.image._load_video_file_bounded", side_effect=fake_load_video
        ):
            text_msgs, images, audio, videos = image_utils.extract_media_from_messages(
                messages
            )

        assert text_msgs == [{"role": "user", "content": "Describe this video"}]
        assert images == []
        assert audio == []
        assert len(videos) == 1
        assert videos[0].frames == "frames"
        assert videos[0].fps == 1.75
        assert videos[0].content_hash == hashlib.sha256(raw_video).hexdigest()
        assert observed["bytes"] == raw_video
        assert observed["field"] == "video_url.url"
        assert not __import__("pathlib").Path(observed["path"]).exists()

    @pytest.mark.parametrize(
        "metadata",
        [
            {"width": float("nan"), "height": 1080, "fps": 30.0, "total_frames": 30},
            {"width": 0, "height": 1080, "fps": 30.0, "total_frames": 30},
            {"width": 4097, "height": 1, "fps": 30.0, "total_frames": 30},
            {"width": 1920, "height": 1080, "fps": 121.0, "total_frames": 30},
            {"width": 1920, "height": 1080, "fps": 30.0, "total_frames": 36_001},
            {"width": 1920, "height": 1080, "fps": 30.0, "total_frames": 9_001},
            {"width": 4096, "height": 2160, "fps": 30.0, "total_frames": 300},
        ],
    )
    def test_video_metadata_limits_fail_closed(self, metadata):
        with pytest.raises(InvalidRequestError):
            image_utils._validated_video_sample_count(**metadata, field="video_url.url")

    def test_video_decode_rejects_output_size_that_disagrees_with_metadata(
        self, monkeypatch, tmp_path
    ):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")

        class FakeCapture:
            def isOpened(self):  # noqa: N802 - mirrors cv2.VideoCapture
                return True

            def get(self, prop):
                return {
                    cv2.CAP_PROP_FRAME_WIDTH: 2,
                    cv2.CAP_PROP_FRAME_HEIGHT: 2,
                    cv2.CAP_PROP_FPS: 2.0,
                    cv2.CAP_PROP_FRAME_COUNT: 4,
                }[prop]

            def set(self, prop, value):
                return True

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())

        def fake_run(command, **kwargs):
            with open(command[-1], "wb") as output_file:
                output_file.write(b"short")
            return type("Result", (), {"returncode": 0})()

        monkeypatch.setattr("subprocess.run", fake_run)
        output_path = tmp_path / "frames.bin"
        with pytest.raises(InvalidRequestError, match="decodable video"):
            image_utils._decode_video_file_to_memmap(
                "ignored.mp4", str(output_path), field="video_url.url"
            )

    def test_video_decode_preallocates_output_without_stacking(
        self, monkeypatch, tmp_path
    ):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")

        class FakeCapture:
            def isOpened(self):  # noqa: N802 - mirrors cv2.VideoCapture
                return True

            def get(self, prop):
                return {
                    cv2.CAP_PROP_FRAME_WIDTH: 2,
                    cv2.CAP_PROP_FRAME_HEIGHT: 2,
                    cv2.CAP_PROP_FPS: 2.0,
                    cv2.CAP_PROP_FRAME_COUNT: 4,
                }[prop]

            def set(self, prop, value):
                return True

            def read(self):
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCapture())
        monkeypatch.setattr(
            np,
            "stack",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("decode must not stack retained frames")
            ),
        )

        def fake_run(command, **kwargs):
            assert command[command.index("-max_alloc") + 1] == str(
                image_utils.MAX_VIDEO_DECODER_ALLOCATION_BYTES
            )
            assert command[command.index("-threads") + 1] == "1"
            assert command[command.index("-frames:v") + 1] == "4"
            assert kwargs["timeout"] == image_utils.VIDEO_DECODE_TIMEOUT_SECONDS
            assert kwargs["stdin"] is __import__("subprocess").DEVNULL
            assert kwargs["stdout"] is __import__("subprocess").DEVNULL
            assert kwargs["stderr"] is __import__("subprocess").DEVNULL
            with open(command[-1], "wb") as output_file:
                output_file.write(bytes(4 * 2 * 2 * 3))
            return type("Result", (), {"returncode": 0})()

        monkeypatch.setattr("subprocess.run", fake_run)

        output_path = tmp_path / "frames.bin"
        shape, fps = image_utils._decode_video_file_to_memmap(
            "ignored.mp4", str(output_path), field="video_url.url"
        )

        assert shape == (4, 2, 2, 3)
        assert fps == 2.0

    def test_video_decode_runs_with_decoder_allocation_cap(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")

        if not __import__("pathlib").Path(image_utils.FFMPEG_BINARY).is_file():
            pytest.skip("fixed ffmpeg binary is unavailable")

        video_path = tmp_path / "tiny.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            2.0,
            (16, 16),
        )
        if not writer.isOpened():
            pytest.skip("OpenCV MJPG writer is unavailable")
        for value in (0, 64, 128, 255):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
        writer.release()

        frames, fps = image_utils._load_video_file_bounded(
            str(video_path), field="video_url.url"
        )

        assert frames.shape == (4, 3, 16, 16)
        assert fps == pytest.approx(2.0)
        assert isinstance(frames, np.memmap)

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/video.mp4",
            "/tmp/video.mp4",
            "data:image/png;base64,AAAA",
            "data:video/mp4;base64,not-valid-base64!!!",
        ],
    )
    def test_video_url_rejects_non_inline_or_malformed_inputs(self, url):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": url}},
                ],
            }
        ]

        with pytest.raises(InvalidRequestError):
            image_utils.extract_media_from_messages(messages)

    def test_rejects_more_than_one_video(self):
        uri = "data:video/mp4;base64," + base64.b64encode(b"video").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                    {"type": "video_url", "video_url": {"url": uri}},
                ],
            }
        ]

        with pytest.raises(InvalidRequestError, match="one video"):
            image_utils.extract_media_from_messages(messages)

    def test_rejects_video_outside_user_message(self):
        uri = "data:video/mp4;base64," + base64.b64encode(b"video").decode()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                ],
            }
        ]

        with pytest.raises(InvalidRequestError, match="user messages"):
            image_utils.extract_media_from_messages(messages)

    def test_rejects_oversized_video_before_decode(self, monkeypatch):
        monkeypatch.setattr(image_utils, "MAX_VIDEO_BYTES", 3)
        uri = "data:video/mp4;base64," + base64.b64encode(b"four").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": uri}},
                ],
            }
        ]

        with pytest.raises(InvalidRequestError, match="exceeds"):
            image_utils.extract_media_from_messages(messages)


# =============================================================================
# Tests: compute_image_hash
# =============================================================================


class TestComputeImageHash:
    """Tests for compute_image_hash()."""

    def test_empty_list_returns_none(self):
        """Empty image list returns None."""
        assert compute_image_hash([]) is None

    def test_single_image_returns_hex_string(self):
        """Single image returns a hex hash string."""
        img = _make_test_image(4, 4, "red")
        result = compute_image_hash([img])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex

    def test_deterministic(self):
        """Same image produces same hash."""
        img1 = _make_test_image(4, 4, "red")
        img2 = _make_test_image(4, 4, "red")
        assert compute_image_hash([img1]) == compute_image_hash([img2])

    def test_different_images_different_hash(self):
        """Different images produce different hashes."""
        img_red = _make_test_image(4, 4, "red")
        img_blue = _make_test_image(4, 4, "blue")
        assert compute_image_hash([img_red]) != compute_image_hash([img_blue])

    def test_order_matters(self):
        """Image order affects the hash."""
        img1 = _make_test_image(4, 4, "red")
        img2 = _make_test_image(4, 4, "blue")
        hash_12 = compute_image_hash([img1, img2])
        hash_21 = compute_image_hash([img2, img1])
        assert hash_12 != hash_21

    def test_multiple_images(self):
        """Multiple images produce a single hash."""
        images = [_make_test_image(4, 4, c) for c in ("red", "green", "blue")]
        result = compute_image_hash(images)
        assert isinstance(result, str)
        assert len(result) == 64


class TestComputePerImageHashes:
    """Tests for compute_per_image_hashes()."""

    def test_returns_one_hash_per_image(self):
        """Returns a list with the same length as the input."""
        images = [_make_test_image(4, 4, c) for c in ("red", "green", "blue")]
        hashes = compute_per_image_hashes(images)
        assert len(hashes) == 3
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes)

    def test_per_image_matches_single_compute(self):
        """Each per-image hash matches compute_image_hash([single_image])."""
        images = [_make_test_image(4, 4, c) for c in ("red", "green")]
        per_hashes = compute_per_image_hashes(images)
        for img, h in zip(images, per_hashes):
            assert h == compute_image_hash([img])

    def test_different_images_different_hashes(self):
        """Different images produce different per-image hashes."""
        images = [_make_test_image(4, 4, "red"), _make_test_image(4, 4, "blue")]
        hashes = compute_per_image_hashes(images)
        assert hashes[0] != hashes[1]

    def test_empty_returns_empty(self):
        """Empty list returns empty list."""
        assert compute_per_image_hashes([]) == []
