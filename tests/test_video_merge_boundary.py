from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _load_video_utils_module(temp_directory: Path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: str(temp_directory)
    sys.modules["folder_paths"] = folder_paths

    package = types.ModuleType("easy_media")
    package.__path__ = []
    utils_package = types.ModuleType("easy_media.utils")
    utils_package.__path__ = []
    media_module = types.ModuleType("easy_media.utils.media")
    media_module.AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac"})
    sys.modules["easy_media"] = package
    sys.modules["easy_media.utils"] = utils_package
    sys.modules["easy_media.utils.media"] = media_module

    path = Path(__file__).parents[1] / "utils" / "video.py"
    spec = importlib.util.spec_from_file_location("easy_media.utils.video", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_video_merge_pads_each_segment_to_its_exact_timeline_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(module, "ffprobe_info", lambda _source: {"has_audio": False})
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = module.merge_video_track_with_ffmpeg(
        [{
            "source": str(source),
            "start_frame": 24,
            "end_frame": 48,
            "source_start_frame": 12,
        }],
        total_length=48,
        frame_rate=24,
        width=320,
        height=180,
    )

    assert output is not None
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "trim=start_frame=12:end_frame=36" in filter_graph
    assert "setpts=PTS-STARTPTS,fps=fps=24:start_time=0" in filter_graph
    assert "tpad=stop_mode=clone:stop_duration=1.0,trim=end_frame=24" in filter_graph
    assert "enable='between(n,24,47)'" in filter_graph


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the decoded-frame boundary regression test",
)
def test_video_merge_clones_the_last_frame_when_source_decodes_two_frames_short(
    tmp_path: Path,
):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "short-blue.mp4"
    create_result = subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24",
            "-frames:v",
            "22",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        capture_output=True,
    )
    assert create_result.returncode == 0, create_result.stderr.decode(errors="replace")

    output = module.merge_video_track_with_ffmpeg(
        [{
            "source": str(source),
            "start_frame": 0,
            "end_frame": 24,
            "source_start_frame": 0,
        }],
        total_length=24,
        frame_rate=24,
        width=64,
        height=64,
    )

    assert output is not None
    decode_result = subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-i",
            output,
            "-vf",
            "select=eq(n\\,23)",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
    )
    assert decode_result.returncode == 0, decode_result.stderr.decode(errors="replace")
    assert decode_result.stdout
    blue_channel = decode_result.stdout[2::3]
    assert sum(blue_channel) / len(blue_channel) > 200


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the fractional-time boundary regression test",
)
def test_video_merge_does_not_expose_black_at_a_fractional_second_boundary(
    tmp_path: Path,
):
    module = _load_video_utils_module(tmp_path)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    sources = []
    for color, frame_count in (("red", 260), ("blue", 2)):
        source = tmp_path / f"{color}.mp4"
        create_result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x64:r=24",
                "-frames:v",
                str(frame_count),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            capture_output=True,
        )
        assert create_result.returncode == 0, create_result.stderr.decode(errors="replace")
        sources.append(source)

    output = module.merge_video_track_with_ffmpeg(
        [
            {
                "source": str(sources[0]),
                "start_frame": 0,
                "end_frame": 260,
                "source_start_frame": 0,
            },
            {
                "source": str(sources[1]),
                "start_frame": 260,
                "end_frame": 262,
                "source_start_frame": 0,
            },
        ],
        total_length=262,
        frame_rate=24,
        width=64,
        height=64,
    )

    assert output is not None
    decode_result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            output,
            "-vf",
            "select=eq(n\\,260)",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
    )
    assert decode_result.returncode == 0, decode_result.stderr.decode(errors="replace")
    assert decode_result.stdout
    blue_channel = decode_result.stdout[2::3]
    assert sum(blue_channel) / len(blue_channel) > 200
