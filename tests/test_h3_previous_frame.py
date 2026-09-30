import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import h3_previous_frame as previous_frame
from test_minimax_node import _load_minimax_node, _h3_project_inputs, _h3_sampling_mode


def test_previous_frame_pins_version_and_rejects_overwritten_source(monkeypatch, tmp_path):
    video = tmp_path / "video_0_1.mp4"
    video.write_bytes(b"video")
    record = {"video": video.name, "sampling_pass": "second"}
    checkpoint = {"sampling_pass": "second"}
    active_version = "1"
    records = {"1": record}
    selected = []

    def generation(name, index, version=None):
        selected.append((index, version))
        resolved = version or active_version
        return tmp_path, {}, resolved, records[resolved] if index == 0 else checkpoint

    monkeypatch.setattr(previous_frame, "project_generation", generation)
    previous_frame.save_tail_image(torch.tensor([[[[1.0, 0.5, 0.0]]]]), tmp_path / "last_frame_0_1.png")
    image, source = previous_frame.load_previous_frame("demo", 1)
    assert image[0, 0, 0].tolist() == pytest.approx([1, 128 / 255, 0])
    checkpoint["previous_frame_source"] = json.loads(source)
    # Resuming must reuse the completed segment's source even if selection changes.
    active_version = "2"
    (tmp_path / "video_0_2.mp4").write_bytes(b"new video")
    records["2"] = {"video": "video_0_2.mp4"}
    restored, restored_source = previous_frame.load_previous_frame("demo", 1, resume=True)
    assert torch.equal(restored, image) and restored_source == source
    assert selected[-1] == (0, "1")
    checkpoint["previous_frame_source"]["revision"] = "overwritten"
    with pytest.raises(ValueError, match="overwritten"):
        previous_frame.load_previous_frame("demo", 1, resume=True)
    active_version = "1"
    record["sampling_pass"] = "first"
    with pytest.raises(ValueError, match="first-pass"):
        previous_frame.load_previous_frame("demo", 1)


def test_old_version_extracts_tail_once_and_preserves_uint8(monkeypatch, tmp_path):
    video = tmp_path / "video_0_1.mp4"
    video.write_bytes(b"video")
    record = {"video": video.name}
    monkeypatch.setattr(previous_frame, "project_generation", lambda *args: (tmp_path, {}, "1", record))
    monkeypatch.setattr(previous_frame, "video_frame_count", lambda path: 73)
    reads = []
    monkeypatch.setattr(previous_frame, "read_video_frames", lambda path, start, count, fps: (
        reads.append((start, count, fps)) or torch.full((1, 1, 1, 3), 128, dtype=torch.uint8)
    ))
    for _ in range(2):
        image, _ = previous_frame.load_previous_frame("demo", 1)
        assert image.mean().item() == pytest.approx(128 / 255)
    assert reads == [(72, 1, None)]
    assert record["last_frame"] == "last_frame_0_1.png"


@pytest.mark.parametrize("fps", [12, 30])
def test_legacy_tail_uses_the_actual_last_frame_at_source_rate(monkeypatch, tmp_path, fps):
    ffmpeg = previous_frame.get_ffmpeg_path()
    if not ffmpeg or not previous_frame.get_ffmpeg_path("ffprobe"):
        pytest.skip("FFmpeg and FFprobe required")
    video = tmp_path / "video_0_1.mp4"
    count = fps * 2
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=red:s=32x32:r={fps}:d=2",
        "-vf", f"drawbox=color=blue:t=fill:enable='eq(n,{count - 1})'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], check=True, capture_output=True)
    record = {"video": video.name, "sampling_pass": "single"}
    monkeypatch.setattr(previous_frame, "project_generation", lambda *args: (tmp_path, {}, "1", record))
    assert previous_frame.video_frame_count(video) == count
    image, _ = previous_frame.load_previous_frame("demo", 1)
    assert image.shape == (1, 32, 32, 3)
    assert image.mean(dim=(0, 1, 2)).argmax().item() == 2  # Only the final frame is blue.
    assert (tmp_path / record["last_frame"]).is_file()


@pytest.mark.parametrize("mode", ["single", "dual", "selflift"])
def test_previous_image_stays_independent_of_context_chain(monkeypatch, mode):
    module = _load_minimax_node(monkeypatch)
    inputs = _h3_project_inputs(sampling_mode=_h3_sampling_mode(mode, upscale_by=[1.0]))
    segments = inputs["tracks_info"][0]["tracks"][0]["segments"]
    for index, continuity in ((1, "shot"), (2, "context")):
        segments.append({"start_frame": index * 120, "end_frame": (index + 1) * 120,
                         "content": {"task_mode": "ref", "continuity_mode": continuity,
                                     "images": [{"source_type": "previous_frame"}], "user_prompt": "dance"}})
    result = module.EasyMultiTrackProject.execute(**inputs)
    nodes = result.expand
    images = [node for node in nodes.values() if node["class_type"] == "easy h3PreviousFrame"]
    assert [node["inputs"]["segment_index"] for node in images] == [1, 2]
    assert not any(node["class_type"] == "easy h3ConditioningCache" and node["inputs"]["segment_index"] in {1, 2}
                   for node in nodes.values())
    artifacts = [node for node in nodes.values() if node["class_type"] == "easy h3ProjectArtifact"]
    assert all("last_frame" in node["inputs"] for node in artifacts)
    if mode == "selflift":
        samplers = [node for node in nodes.values() if node["class_type"] == "easy minimaxH3SelfLiftSampler"]
        assert "low_context_latent" not in samplers[1]["inputs"]
        assert samplers[1]["inputs"]["rho"] == 0
        assert "low_context_latent" in samplers[2]["inputs"]
        assert samplers[2]["inputs"]["rho"] == 0.1
    else:
        contexts = [node for node in nodes.values() if node["class_type"] == "easy MiniMaxH3MotionContextHard"]
        assert len(contexts) == 1
