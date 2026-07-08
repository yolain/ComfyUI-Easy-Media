from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import speech


def _install_fake_torch(monkeypatch, *, cuda_available: bool = False, mps_available: bool = False):
    class FakeCuda:
        @staticmethod
        def is_available():
            return cuda_available

    class FakeMpsBackend:
        @staticmethod
        def is_available():
            return mps_available

    class FakeBackends:
        mps = FakeMpsBackend()

    fake_torch = types.SimpleNamespace(cuda=FakeCuda(), backends=FakeBackends())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_requires_voxcpm_dependency(monkeypatch):
    monkeypatch.setattr(speech.importlib.util, "find_spec", lambda name: None if name == "voxcpm" else object())

    assert speech.missing_speech_dependencies() == ["voxcpm"]


def test_requires_voxcpm2_model_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(speech.folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(speech, "notify_missing_model", lambda model_name: {"name": model_name})

    with pytest.raises(speech.MissingEasyMediaModelError):
        speech.require_voxcpm2_model_dir()


def test_generate_voxcpm2_speech_passes_cfg_steps_normalize_and_reference(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"ref")
    monkeypatch.setattr(speech.folder_paths, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(speech.folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(speech, "missing_speech_dependencies", lambda: [])
    _install_fake_torch(monkeypatch)

    calls = {}
    cleaned = []

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return cls()

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return 16000, [0.0, 0.1]

    voxcpm_module = types.ModuleType("voxcpm")
    voxcpm_module.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_module)
    monkeypatch.setattr(speech, "_write_audio_file", lambda path, audio, sample_rate: path.write_bytes(b"wav"))
    monkeypatch.setattr(speech, "cleanup_model_memory", lambda *models: cleaned.extend(models))
    monkeypatch.setattr(speech, "_seed_voxcpm_generation", lambda: calls.setdefault("seeded", True))

    result = speech.generate_voxcpm2_speech(
        text="字幕123",
        prompt="四川话",
        cfg=2.3,
        steps=12,
        reference_audio_path=reference,
    )

    assert calls["model_path"] == str(model_dir)
    assert calls["model_kwargs"] == {
        "device": "cpu",
        "optimize": False,
    }
    assert calls["generate"] == {
        "text": "(四川话)字幕123",
        "cfg_value": 2.3,
        "inference_timesteps": 12,
        "normalize": True,
        "reference_wav_path": str(reference),
    }
    assert calls["seeded"] is True
    assert len(cleaned) == 1
    assert isinstance(cleaned[0], FakeVoxCPM)
    assert result.source_type == "output"
    assert result.file_path.startswith("easy_media/字幕123_")
    assert Path(result.absolute_path).is_file()


def test_generate_voxcpm2_speech_accepts_raw_waveform_result(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(speech.folder_paths, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(speech.folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(speech, "missing_speech_dependencies", lambda: [])
    _install_fake_torch(monkeypatch)

    calls = {}

    class FakeTtsModel:
        sample_rate = 24000

    class FakeVoxCPM:
        tts_model = FakeTtsModel()

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def generate(self, **kwargs):
            return [0.0, 0.1, 0.2]

    def write_audio_file(path, audio, sample_rate):
        calls["audio"] = audio
        calls["sample_rate"] = sample_rate
        path.write_bytes(b"wav")

    voxcpm_module = types.ModuleType("voxcpm")
    voxcpm_module.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_module)
    monkeypatch.setattr(speech, "_write_audio_file", write_audio_file)
    monkeypatch.setattr(speech, "cleanup_model_memory", lambda *models: None)
    monkeypatch.setattr(speech, "_seed_voxcpm_generation", lambda: 123)

    result = speech.generate_voxcpm2_speech(
        text="字幕",
        prompt="",
        cfg=2.0,
        steps=10,
    )

    assert calls["audio"] == [0.0, 0.1, 0.2]
    assert calls["sample_rate"] == 24000
    assert result.duration == pytest.approx(3 / 24000)


def test_generate_voxcpm2_speech_cleans_model_memory_on_error(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(speech.folder_paths, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(speech, "missing_speech_dependencies", lambda: [])
    cleaned = []

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def generate(self, **kwargs):
            raise RuntimeError("boom")

    voxcpm_module = types.ModuleType("voxcpm")
    voxcpm_module.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_module)
    monkeypatch.setattr(speech, "cleanup_model_memory", lambda *models: cleaned.extend(models), raising=False)

    with pytest.raises(RuntimeError, match="boom"):
        speech.generate_voxcpm2_speech(
            text="字幕",
            prompt="",
            cfg=2.0,
            steps=10,
        )

    assert len(cleaned) == 1
    assert isinstance(cleaned[0], FakeVoxCPM)


def test_generate_voxcpm2_speech_disables_warmup_without_cuda_or_mps(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(speech.folder_paths, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(speech.folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(speech, "missing_speech_dependencies", lambda: [])

    calls = {}

    _install_fake_torch(monkeypatch)

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return cls()

        def generate(self, **kwargs):
            return 16000, [0.0]

    voxcpm_module = types.ModuleType("voxcpm")
    voxcpm_module.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_module)
    monkeypatch.setattr(speech, "_write_audio_file", lambda path, audio, sample_rate: path.write_bytes(b"wav"))
    monkeypatch.setattr(speech, "cleanup_model_memory", lambda *models: None, raising=False)

    speech.generate_voxcpm2_speech(
        text="字幕",
        prompt="",
        cfg=2.0,
        steps=10,
    )

    assert calls["model_path"] == str(model_dir)
    assert calls["model_kwargs"] == {
        "device": "cpu",
        "optimize": False,
    }


def test_generate_voxcpm2_speech_uses_subtitle_text_when_prompt_is_blank(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(speech.folder_paths, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(speech.folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(speech, "missing_speech_dependencies", lambda: [])
    _install_fake_torch(monkeypatch)

    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return 16000, [0.0]

    voxcpm_module = types.ModuleType("voxcpm")
    voxcpm_module.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_module)
    monkeypatch.setattr(speech, "_write_audio_file", lambda path, audio, sample_rate: path.write_bytes(b"wav"))
    monkeypatch.setattr(speech, "cleanup_model_memory", lambda *models: None, raising=False)
    monkeypatch.setattr(speech, "_seed_voxcpm_generation", lambda: 123)

    speech.generate_voxcpm2_speech(
        text="字幕",
        prompt="  ",
        cfg=2.0,
        steps=10,
    )

    assert calls["generate"]["text"] == "字幕"
    assert "prompt_text" not in calls["generate"]
    assert "prompt_wav_path" not in calls["generate"]
