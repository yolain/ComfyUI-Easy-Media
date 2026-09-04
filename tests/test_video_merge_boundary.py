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
    assert (
        "setpts=PTS-STARTPTS,fps=fps=24:start_time=0,"
        "trim=start_frame=12:end_frame=36,setpts=PTS-STARTPTS"
    ) in filter_graph
    assert "tpad=stop_mode=clone:stop_duration=1.0,trim=end_frame=24" in filter_graph
    assert "setpts=PTS-STARTPTS+24" in filter_graph
    assert "enable='between(round(t*24),24,47)'" in filter_graph


def test_video_merge_uses_sample_boundaries_only_for_locked_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(module, "ffprobe_info", lambda _source: {"has_audio": True})
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = module.merge_video_track_with_ffmpeg(
        [
            {
                "source": str(source),
                "start_frame": 0,
                "end_frame": 209,
                "source_start_frame": 0,
                "audio_locked": True,
            },
            {
                "source": str(source),
                "start_frame": 209,
                "end_frame": 350,
                "source_start_frame": 0,
                "audio_locked": False,
            },
        ],
        total_length=350,
        frame_rate=24,
        width=320,
        height=180,
    )

    assert output is not None
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "aresample=44100:first_pts=0" in filter_graph
    assert "atrim=start_sample=0:end_sample=384038" in filter_graph
    assert "adelay=0S:all=1" in filter_graph
    assert "atrim=start=0.0:duration=5.875" in filter_graph
    assert "adelay=8708:all=1" in filter_graph


def test_video_merge_uses_separate_original_audio_for_locked_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_video_utils_module(tmp_path)
    source = tmp_path / "source.mp4"
    original_audio = tmp_path / "locked.wav"
    source.write_bytes(b"video")
    original_audio.write_bytes(b"audio")
    monkeypatch.setattr(module, "get_ffmpeg_path", lambda _name="ffmpeg": "ffmpeg")
    monkeypatch.setattr(module, "ffprobe_info", lambda _source: {"has_audio": True})
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = module.merge_video_track_with_ffmpeg(
        [{
            "source": str(source),
            "audio_source": str(original_audio),
            "start_frame": 0,
            "end_frame": 48,
            "source_start_frame": 12,
            "audio_locked": True,
        }],
        total_length=48,
        frame_rate=24,
        width=320,
        height=180,
    )

    assert output is not None
    command = commands[0]
    assert command[command.index(str(source)) + 1:command.index("-filter_complex")] == [
        "-i",
        str(original_audio),
    ]
    filter_graph = command[command.index("-filter_complex") + 1]
    assert (
        "[2:v]setpts=PTS-STARTPTS,fps=fps=24:start_time=0,"
        "trim=start_frame=12:end_frame=60"
    ) in filter_graph
    assert "[3:a]aresample=44100:first_pts=0" in filter_graph
    assert "atrim=start_sample=22050:end_sample=110250" in filter_graph


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
    reason="FFmpeg is required for the source-frame-rate regression test",
)
def test_video_merge_normalizes_source_rate_before_trimming_timeline_frames(
    tmp_path: Path,
):
    module = _load_video_utils_module(tmp_path)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "five-seconds-at-30fps.mp4"
    create_result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=30:d=4",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=30:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-frames:v",
            "150",
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
        [{"source": str(source), "start_frame": 0, "end_frame": 120}],
        total_length=120,
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
            "select=eq(n\\,108)",
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
    red_channel = decode_result.stdout[0::3]
    blue_channel = decode_result.stdout[2::3]
    assert sum(blue_channel) / len(blue_channel) > 200
    assert sum(red_channel) / len(red_channel) < 30


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the full-timeline regression test",
)
@pytest.mark.parametrize("frame_rate", [24, 30, 30000 / 1001])
@pytest.mark.parametrize("layout", ["sequential", "gaps-and-overlap"])
def test_video_merge_preserves_every_frame_of_later_segments(
    tmp_path: Path,
    frame_rate: float,
    layout: str,
) -> None:
    module = _load_video_utils_module(tmp_path)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    if layout == "sequential":
        clips = [(0, 24, "red"), (24, 48, "blue"), (48, 72, "lime")]
        total_length = 72
    else:
        clips = [(3, 29, "red"), (29, 53, "blue"), (55, 66, "lime"), (60, 64, "yellow")]
        total_length = 70
    colors = {
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "lime": (0, 255, 0),
        "yellow": (255, 255, 0),
    }
    expected = ["black"] * total_length
    segments = []
    for start, end, color in clips:
        source = tmp_path / f"{color}.mp4"
        created = subprocess.run(
            [
                ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=64x64:r={frame_rate}",
                "-frames:v", str(end - start), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(source),
            ],
            capture_output=True,
        )
        assert created.returncode == 0, created.stderr.decode(errors="replace")
        segments.append({"source": str(source), "start_frame": start, "end_frame": end})
        expected[start:end] = [color] * (end - start)

    output = module.merge_video_track_with_ffmpeg(
        segments, total_length, frame_rate, 64, 64,
    )
    assert output is not None
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", output, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    assert decoded.returncode == 0, decoded.stderr.decode(errors="replace")
    frame_bytes = 64 * 64 * 3
    assert len(decoded.stdout) == total_length * frame_bytes
    for index, color in enumerate(expected):
        frame = decoded.stdout[index * frame_bytes:(index + 1) * frame_bytes]
        actual = tuple(sum(frame[channel::3]) / (64 * 64) for channel in range(3))
        assert actual == pytest.approx(colors[color], abs=12), (
            f"Frame {index}: expected {color}, got RGB {actual}"
        )


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
