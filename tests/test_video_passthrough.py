from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.video import passthrough_video_media


class FakeVideo:
    def __init__(self, frames: int, fps: int, audio=None):
        self.components = SimpleNamespace(
            images=torch.arange(frames).float().view(frames, 1, 1, 1).expand(-1, 2, 2, 3),
            frame_rate=Fraction(fps),
            audio=audio,
        )

    def get_components(self):
        return self.components


def test_passthrough_uses_first_video_and_complete_timeline():
    first = FakeVideo(60, 30)
    second = FakeVideo(48, 24)
    images, audio = passthrough_video_media([first, second], 48, 24)
    assert images.shape[0] == 48
    assert images[:, 0, 0, 0].tolist() == [int(i * 30 / 24) for i in range(48)]
    assert audio["waveform"].shape[-1] == 2 * 44100


def test_passthrough_rejects_missing_or_short_video():
    with pytest.raises(ValueError, match="index 0"):
        passthrough_video_media([None, FakeVideo(48, 24)], 48, 24)
    with pytest.raises(ValueError, match="complete task timeline"):
        passthrough_video_media([FakeVideo(47, 24)], 48, 24)
