import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_module():
    path = Path(__file__).parents[1] / "utils" / "h3_audio_remix.py"
    spec = importlib.util.spec_from_file_location("h3_audio_remix_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Nested:
    is_nested = True

    def __init__(self, *streams):
        self.tensors = streams

    def unbind(self):
        return self.tensors


def _latent(audio_mask):
    video = torch.full((1, 24, 1, 1, 1), 2.0)
    audio = torch.full((1, 32, 2, 1), 2.0)
    return {
        "samples": _Nested(video, audio),
        "noise_mask": _Nested(torch.full_like(video, 0.25), torch.full_like(audio, audio_mask)),
    }


def _pack(nested):
    return torch.cat([stream.reshape(1, 1, -1) for stream in nested.unbind()], dim=-1)


class _Inpaint:
    """Comfy's two inpaint blends around a deterministic H3 velocity oracle."""

    def __init__(self, latent, sigmas):
        self.latent_image = _pack(latent["samples"])
        self.noise = torch.full_like(self.latent_image, 8.0)
        self.sigmas = sigmas
        self.mask = _pack(latent["noise_mask"])
        self.calls = []
        self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(
            scale_latent_inpaint=lambda **kwargs: kwargs["latent_image"],
        ))

    def __call__(self, x, sigma, denoise_mask, model_options=None, seed=None):
        options = model_options or {}
        if "denoise_mask_function" in options:
            denoise_mask = options["denoise_mask_function"](
                sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas},
            )
        self.calls.append((denoise_mask.clone(), seed))
        model_x = x * denoise_mask + self.latent_image * (1.0 - denoise_mask)
        # H3 already scales the flow velocity by its conditioning mask.
        velocity = (8.0 - 10.0) * self.mask
        denoised = model_x - sigma * velocity
        return denoised * denoise_mask + self.latent_image * (1.0 - denoise_mask)


def _euler(model, x, sigmas, extra_args=None, callback=None):
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = model(x, sigma.reshape(1), **(extra_args or {}))
        x = x + (x - denoised) / sigma * (sigmas[index + 1] - sigma)
        if callback:
            callback(index, denoised)
    return x


@pytest.mark.parametrize("strength", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("steps", [4, 20, 50])
def test_soft_audio_retains_generated_residual_independent_of_step_count(strength, steps):
    module = _load_module()
    latent = _latent(1.0 - strength)
    sigmas = torch.linspace(1, 0, steps + 1)
    native = SimpleNamespace(sampler_function=_euler)
    corrected = module.wrap_h3_audio_remix_sampler(native, latent)
    model = _Inpaint(latent, sigmas)
    mask = model.mask.clone()
    kwargs = {"extra_args": {"denoise_mask": mask, "seed": 42}}
    baseline = native.sampler_function(_Inpaint(latent, sigmas), model.noise.clone(), sigmas, **kwargs)
    output = corrected.sampler_function(model, model.noise.clone(), sigmas, **kwargs)
    expected = 2.0 * strength + 10.0 * (1.0 - strength)
    assert torch.allclose(output[..., 24:], torch.full_like(output[..., 24:], expected), atol=1e-5)
    assert not torch.allclose(baseline[..., 24:], output[..., 24:], atol=0.01)
    assert torch.equal(output[..., :24], baseline[..., :24])
    assert torch.equal(mask, model.mask)
    assert native.sampler_function is _euler
    assert all(seed == 42 for _, seed in model.calls)


@pytest.mark.parametrize("audio_mask", [0.0, 1.0])
def test_hard_lock_and_full_generation_keep_the_original_sampler(audio_mask):
    sampler = SimpleNamespace(sampler_function=_euler)
    assert _load_module().wrap_h3_audio_remix_sampler(sampler, _latent(audio_mask)) is sampler


def test_dynamic_video_mask_runs_once_per_step_and_callbacks_are_forwarded():
    latent = _latent(0.5)
    sigmas = torch.linspace(1, 0, 5)
    model = _Inpaint(latent, sigmas)
    masks = []
    callbacks = []

    def dynamic_mask(sigma, mask, extra_options):
        assert extra_options["sigmas"] is sigmas
        masks.append(float(sigma[0]))
        updated = mask.clone()
        updated[..., :24] = 0
        return updated

    options = {"denoise_mask_function": dynamic_mask}
    sampler = _load_module().wrap_h3_audio_remix_sampler(SimpleNamespace(sampler_function=_euler), latent)
    result = sampler.sampler_function(model, model.noise.clone(), sigmas,
        extra_args={"denoise_mask": model.mask, "model_options": options},
        callback=lambda step, _: callbacks.append(step))
    assert len(masks) == 4
    assert callbacks == [0, 1, 2, 3]
    assert "denoise_mask_function" in options
    assert torch.equal(result[..., :24], torch.full_like(result[..., :24], 2.0))
    assert all(torch.all(mask[..., 24:] == 1) for mask, _ in model.calls)


def test_audio_context_anchors_stay_hard_locked():
    latent = _latent(0.5)
    latent["noise_mask"].tensors[1][:, :16] = 0
    sigmas = torch.linspace(1, 0, 5)
    model = _Inpaint(latent, sigmas)
    sampler = _load_module().wrap_h3_audio_remix_sampler(SimpleNamespace(sampler_function=_euler), latent)
    output = sampler.sampler_function(model, model.noise.clone(), sigmas, extra_args={"denoise_mask": model.mask})
    assert torch.all(output[..., 24:56] == 2)
    assert torch.allclose(output[..., 56:], torch.full_like(output[..., 56:], 6))


def test_soft_remix_with_the_installed_comfy_inpaint_adapter():
    # Exercise Comfy's actual adapter without importing its GPU/model runtime.
    source = Path(__file__).resolve().parents[3] / "comfy" / "samplers.py"
    if not source.is_file():
        pytest.skip("ComfyUI source is not installed alongside this node pack")
    tree = ast.parse(source.read_text())
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KSamplerX0Inpaint")
    namespace = {}
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), str(source), "exec"), namespace)
    latent = _latent(0.9)
    sigmas = torch.linspace(1, 0, 21)
    source_latent = _pack(latent["samples"])
    mask = _pack(latent["noise_mask"])

    class Guider:
        inner_model = SimpleNamespace(scale_latent_inpaint=lambda **kwargs: kwargs["latent_image"])

        def __call__(self, x, sigma, model_options, seed):
            return x - sigma * mask * (8.0 - 10.0)

    model = namespace["KSamplerX0Inpaint"](Guider(), sigmas)
    model.latent_image = source_latent
    model.noise = torch.full_like(source_latent, 8.0)
    sampler = _load_module().wrap_h3_audio_remix_sampler(SimpleNamespace(sampler_function=_euler), latent)
    output = sampler.sampler_function(model, model.noise.clone(), sigmas, extra_args={"denoise_mask": mask})
    assert torch.allclose(output[..., 24:], torch.full_like(output[..., 24:], 9.2), atol=1e-5)


def test_stochastic_sampler_scales_only_soft_audio_noise():
    latent = _latent(0.5)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    model = _Inpaint(latent, sigmas)
    supplied_noise = torch.ones_like(model.noise)

    def stochastic(model, x, sigmas, extra_args=None, noise_sampler=None):
        return noise_sampler(sigmas[0], sigmas[1])

    sampler = _load_module().wrap_h3_audio_remix_sampler(SimpleNamespace(sampler_function=stochastic), latent)
    result = sampler.sampler_function(model, model.noise, sigmas,
        extra_args={"denoise_mask": model.mask}, noise_sampler=lambda *_: supplied_noise)
    assert torch.all(result[..., :24] == 1)
    assert torch.all(result[..., 24:] == 0.5)
    assert torch.all(supplied_noise == 1)
