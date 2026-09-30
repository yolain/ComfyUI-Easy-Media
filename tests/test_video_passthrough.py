from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.minimax import h3_phase_aligned_video_suffix
from utils.video import (
    ffprobe_info,
    get_ffmpeg_path,
    stage_passthrough_video_media,
)
import utils.video as video_utils


class FileVideo:
    def __init__(self, path: Path):
        self.path = path

    def get_stream_source(self):
        return str(self.path)

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_components(self):
        raise AssertionError("File-backed passthrough must not decode the full VIDEO")


class TrimmedFileVideo(FileVideo):
    def __init__(self, path: Path):
        super().__init__(path)
        self.saved = False

    def get_active_trim_window(self):
        return 1.0, 2.0

    def save_to(self, output_path):
        self.saved = True
        subprocess.run(
            [get_ffmpeg_path("ffmpeg"), "-y", "-v", "error", "-ss", "1",
             "-i", str(self.path), "-t", "2", "-c", "copy", str(output_path)],
            check=True, capture_output=True, timeout=30,
        )


def test_file_passthrough_stages_with_ffmpeg_and_decodes_only_context_tail(tmp_path):
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is unavailable")
    source = tmp_path / "source.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x48:rate=24:duration=3", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=44100:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(source)],
        check=True, capture_output=True, timeout=30,
    )
    destination = tmp_path / "staged.mp4"
    path, images, audio = stage_passthrough_video_media(
        FileVideo(source), 72, 24.0, 64, 48, destination,
    )
    assert path == str(destination)
    assert ffprobe_info(path)["frame_count"] == 72
    assert images is not None and images.shape == (22, 48, 64, 3)
    assert audio is not None and audio["waveform"].shape == (1, 2, 40425)
    assert audio["waveform"].abs().mean().item() > 0.001
    full_decode = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True, timeout=30,
    )
    full_images = torch.frombuffer(
        bytearray(full_decode.stdout), dtype=torch.uint8,
    ).reshape(72, 48, 64, 3).to(torch.float32) / 255.0
    assert torch.equal(images, h3_phase_aligned_video_suffix(full_images, 22))

    with pytest.raises(ValueError, match="complete task timeline"):
        stage_passthrough_video_media(
            FileVideo(source), 96, 24.0, 64, 48,
            tmp_path / "too-long.mp4",
        )


def test_trimmed_file_passthrough_uses_active_video_window(tmp_path, monkeypatch):
    ffmpeg = get_ffmpeg_path("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is unavailable")
    monkeypatch.setattr(video_utils.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    source = tmp_path / "source.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x48:rate=24:duration=3", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True, timeout=30,
    )
    video = TrimmedFileVideo(source)
    path, images, audio = stage_passthrough_video_media(
        video, 48, 24.0, 64, 48, tmp_path / "trimmed.mp4",
    )
    assert video.saved
    assert ffprobe_info(path)["frame_count"] == 48
    assert images.shape == (22, 48, 64, 3)
    assert audio["waveform"].shape == (1, 2, 40425)
    assert ffprobe_info(path)["has_audio"] is True


def test_passthrough_without_reference_stages_black_video_and_silence(tmp_path):
    if get_ffmpeg_path("ffmpeg") is None:
        pytest.skip("FFmpeg is unavailable")
    path, images, audio = stage_passthrough_video_media(
        None, 48, 24.0, 64, 48, tmp_path / "black.mp4",
    )
    info = ffprobe_info(path)
    assert info["frame_count"] == 48
    assert (info["width"], info["height"]) == (64, 48)
    assert info["has_audio"] is True
    assert images.shape == (22, 48, 64, 3)
    assert images.max().item() == 0
    assert audio["waveform"].shape == (1, 2, 40425)
    assert audio["waveform"].abs().max().item() == 0
