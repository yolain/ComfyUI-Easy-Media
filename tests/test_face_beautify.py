import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('beauty_under_test', Path(__file__).parents[1] / 'utils/face_beautify.py')
beauty = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beauty)


def geometry():
    return {'handles': [{'center': [.5, .5], 'radius': .25, 'key': 'eyes', 'delta': [0, 0], 'scale': .22}],
            'regions': [{'center': [.5, .5], 'radii': [.3, .4]}], 'axis': [1, 0], 'blur': .01}


def test_preset_round_trip_and_limits():
    parsed = beauty.parse_beauty_preset('{"parameters":{"eyes":2,"smooth":-1}}')
    assert parsed['parameters']['eyes'] == 1
    assert parsed['parameters']['smooth'] == 0
    assert beauty.parse_beauty_preset(json.dumps(parsed)) == parsed


@pytest.mark.parametrize('value', ['[]', '{', '{"version":2}', '{"version":true}', '{"parameters":{"eyes":true}}', '{"parameters":{"eyes":NaN}}', '{"parameters":null}'])
def test_invalid_presets_fail_explicitly(value):
    with pytest.raises(ValueError):
        beauty.parse_beauty_preset(value)


def test_neutral_and_no_face_are_exact_identity():
    image = torch.rand(65, 81, 4)
    params = beauty.parse_beauty_preset('{}')['parameters']
    assert torch.equal(beauty.apply_face_beauty(image, params, geometry()), image)
    params['eyes'] = 1
    assert torch.equal(beauty.apply_face_beauty(image, params, {'handles': []}), image)


def test_warp_changes_inside_and_preserves_outside_radius():
    x = torch.linspace(0, 1, 64)
    image = x[None, :, None].expand(64, 64, 3).clone()
    params = beauty.parse_beauty_preset('{"parameters":{"eyes":1}}')['parameters']
    result = beauty.apply_face_beauty(image, params, geometry())
    assert (result[32, 38] - image[32, 38]).abs().max() > .005
    assert torch.allclose(result[:8], image[:8], atol=1e-6)


def test_skin_adjustments_exclude_eye_and_background_and_preserve_alpha():
    image = torch.full((64, 64, 4), .4)
    image[..., 3] = .7
    geo = geometry()
    geo['regions'].append({'center': [.5, .5], 'radii': [.1, .1]})
    params = beauty.parse_beauty_preset('{"parameters":{"whiten":1}}')['parameters']
    result = beauty.apply_face_beauty(image, params, geo)
    assert torch.allclose(result[32, 32], image[32, 32], atol=1e-6)
    assert torch.allclose(result[0], image[0], atol=1e-6)
    assert result[24, 24, 0] > image[24, 24, 0]
    assert torch.allclose(result[..., 3], image[..., 3], atol=1e-6)


def test_strip_boundaries_remain_continuous():
    image = torch.full((520, 40, 3), .3)
    params = beauty.parse_beauty_preset('{"parameters":{"smooth":1,"brightness":1}}')['parameters']
    out = beauty.apply_face_beauty(image, params, geometry())
    assert out.shape == image.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out[255], out[256], atol=1e-6)


def test_source_path_rejects_traversal_and_symlinks(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    root = tmp_path / 'input'; root.mkdir()
    outside = tmp_path / 'outside.png'; outside.write_bytes(b'x')
    (root / 'link.png').symlink_to(outside)
    monkeypatch.setitem(sys.modules, 'folder_paths', SimpleNamespace(
        get_input_directory=lambda: str(root), get_output_directory=lambda: str(root), get_temp_directory=lambda: str(root)))
    for path in ('../outside.png', str(outside), 'link.png'):
        with pytest.raises(ValueError):
            beauty.beauty_source_path({'source_type': 'input', 'file_path': path})


def test_real_geometry_uses_float32_sampling_grid():
    points = [[.5, .5] for _ in range(478)]
    for index, point in {234: [.25, .4], 454: [.75, .4], 10: [.5, .1], 152: [.5, .9]}.items():
        points[index] = point
    geo = beauty.face_geometry(points, 1.5)
    image = torch.rand(60, 40, 3)
    params = beauty.parse_beauty_preset('{"parameters":{"slim_face":.5,"smooth":.5}}'.replace(':.', ':0.'))['parameters']
    assert beauty.apply_face_beauty(image, params, geo).dtype == torch.float32


def load_node(monkeypatch):
    import sys
    import types
    root = Path(__file__).parents[1]
    for name in ('beauty_package', 'beauty_package.nodes', 'beauty_package.utils'):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'beauty_package.utils.face_beautify', beauty)
    api = types.ModuleType('comfy_api.latest')
    api.io = types.SimpleNamespace(ComfyNode=object, NodeOutput=lambda *args, **kwargs: types.SimpleNamespace(args=args, **kwargs))
    monkeypatch.setitem(sys.modules, 'comfy_api', types.ModuleType('comfy_api'))
    monkeypatch.setitem(sys.modules, 'comfy_api.latest', api)
    spec = importlib.util.spec_from_file_location('beauty_package.nodes.face_beautify', root / 'nodes/face_beautify.py')
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    return node


def test_connected_image_overrides_picker_and_processes_whole_batch(monkeypatch):
    node = load_node(monkeypatch)
    monkeypatch.setattr(beauty, 'detect_face', lambda frame: [])
    monkeypatch.setattr(node, 'beauty_preview', lambda frame, points: {'face_found': False})
    monkeypatch.setattr(node, 'load_beauty_source', lambda _: pytest.fail('Connected IMAGE should override picker'))
    image = torch.rand(3, 32, 48, 3)
    preset = json.dumps({'version': 1, 'parameters': {'eyes': .5}, 'source': {'source_type': 'input', 'file_path': 'missing.png'}})
    result = node.EasyFaceBeautify.execute(preset, image)
    assert torch.equal(result.args[0], image)
    assert json.loads(result.args[1])['parameters']['eyes'] == .5
    assert result.ui['face_beauty'][0]['input_connected'] is True


def test_selected_image_fingerprint_changes_when_file_is_overwritten(tmp_path, monkeypatch):
    node = load_node(monkeypatch)
    path = tmp_path / 'portrait.png'
    monkeypatch.setattr(node, 'beauty_source_path', lambda _: path)
    preset = json.dumps({'source': {'source_type': 'input', 'file_path': 'portrait.png'}})
    path.write_bytes(b'first-image')
    first = node.EasyFaceBeautify.fingerprint_inputs(preset)
    path.write_bytes(b'other-image')
    assert node.EasyFaceBeautify.fingerprint_inputs(preset) != first


def test_selecting_portrait_can_override_connected_image(monkeypatch):
    node = load_node(monkeypatch)
    selected = torch.rand(1, 20, 24, 3)
    monkeypatch.setattr(beauty, 'detect_face', lambda frame: [])
    monkeypatch.setattr(node, 'beauty_preview', lambda frame, points: {'face_found': False})
    monkeypatch.setattr(node, 'load_beauty_source', lambda source: selected)
    preset = json.dumps({'source_mode': 'picker', 'source': {'source_type': 'input', 'file_path': 'selected.png'}})
    result = node.EasyFaceBeautify.execute(preset, torch.zeros(2, 32, 48, 3))
    assert torch.equal(result.args[0], selected)
    assert result.ui['face_beauty'][0]['input_connected'] is False
    assert json.loads(result.args[1])['source_mode'] == 'picker'


def makeup_geometry():
    geo = geometry()
    geo['features'] = {
        'eyes': [[[.25,.25],[.4,.25],[.4,.35],[.25,.35]], [[.6,.25],[.75,.25],[.75,.35],[.6,.35]]],
        'lip_outer': [[.3,.6],[.7,.6],[.7,.8],[.3,.8]],
        'lip_inner': [[.4,.65],[.6,.65],[.6,.75],[.4,.75]],
    }
    geo['accents'] = [
        {'center': [.32,.42], 'radii': [.15,.07]}, {'center': [.68,.42], 'radii': [.15,.07]},
        {'center': [.3,.5], 'radii': [.16,.12]}, {'center': [.7,.5], 'radii': [.16,.12]},
    ]
    return geo


@pytest.mark.parametrize('key,point', [('eye_bright',(19,20)), ('under_eye',(27,20)), ('lip_tint',(40,22)), ('blush',(32,20))])
def test_makeup_changes_target_only_and_preserves_alpha(key, point):
    image = torch.full((64,64,4), .4)
    image[...,3] = .7
    params = beauty.parse_beauty_preset(json.dumps({'parameters': {key:1}}))['parameters']
    result = beauty.apply_face_beauty(image, params, makeup_geometry())
    assert (result[point[0],point[1],:3] - image[point[0],point[1],:3]).abs().max() > .005
    assert torch.allclose(result[:5],image[:5],atol=1e-6)
    assert torch.allclose(result[...,3],image[...,3],atol=1e-6)


def test_lip_tint_preserves_inner_mouth_and_teeth():
    image = torch.full((64,64,3), .6)
    params = beauty.parse_beauty_preset('{"parameters":{"lip_tint":1}}')['parameters']
    result = beauty.apply_face_beauty(image, params, makeup_geometry())
    assert torch.allclose(result[44,32],image[44,32],atol=1e-6)
    assert not torch.allclose(result[40,22],image[40,22])


def test_old_presets_fill_makeup_defaults_and_colors_round_trip():
    old = beauty.parse_beauty_preset('{"parameters":{"eyes":0.5}}')
    for key in ('eye_bright','under_eye','lip_tint','blush'):
        assert old['parameters'][key] == 0
    assert old['colors'] == beauty.DEFAULT_BEAUTY_COLORS
    custom = beauty.parse_beauty_preset('{"colors":{"lip":"#AA3355","blush":"#cc8877"}}')
    assert custom['colors']['lip'] == '#aa3355'
    assert beauty.parse_beauty_preset(json.dumps(custom)) == custom
    with pytest.raises(ValueError):
        beauty.parse_beauty_preset('{"colors":{"lip":"red"}}')


def test_custom_lip_color_affects_output():
    image = torch.full((64,64,3), .6)
    params = beauty.parse_beauty_preset('{"parameters":{"lip_tint":1}}')['parameters']
    red = beauty.apply_face_beauty(image, params, makeup_geometry(), {'lip':'#ff0000','blush':'#ff9f80'})
    blue = beauty.apply_face_beauty(image, params, makeup_geometry(), {'lip':'#0000ff','blush':'#ff9f80'})
    assert red[40,22,0] > blue[40,22,0]
    assert blue[40,22,2] > red[40,22,2]


def test_numeric_schema_uses_native_widgets_and_optional_preset_socket(monkeypatch):
    from types import SimpleNamespace
    node = load_node(monkeypatch)
    def field(name, **options):
        return SimpleNamespace(id=name, **options)
    node.io.Schema = lambda **options: SimpleNamespace(**options)
    node.io.NumberDisplay = SimpleNamespace(number='number')
    for kind in ('Image', 'Float', 'Int', 'String'):
        setattr(node.io, kind, SimpleNamespace(Input=field, Output=field))
    schema = node.EasyFaceBeautifyStatic.define_schema()
    assert schema.is_output_node is False
    inputs = {item.id: item for item in schema.inputs}
    assert inputs['preset'].optional is True
    assert inputs['preset'].force_input is True
    assert inputs['eyes'].display_mode == 'number'
    assert inputs['eyes'].min == -1 and inputs['eyes'].max == 1
    assert inputs['eye_bright'].min == 0
    assert inputs['lip_color'].default == '#c4485f'
    assert inputs['blush_color'].default == '#ff9f80'
    assert 'lip_red' not in inputs
    assert len(schema.inputs) == 61
    # New controls follow the existing color widgets, preserving old workflows.
    assert [item.id for item in schema.inputs[:16]] == ['image', *beauty.LEGACY_BEAUTY_PARAMETERS, 'lip_color', 'blush_color']
    assert inputs['pointed_chin'].default == 0


def test_numeric_neutral_node_returns_image_without_detection_or_ui(monkeypatch):
    node = load_node(monkeypatch)
    monkeypatch.setattr(beauty, 'detect_face', lambda _: pytest.fail('Neutral numeric node needs no detection'))
    monkeypatch.setattr(node, 'beauty_preview', lambda *_: pytest.fail('Numeric node must not create previews'))
    image = torch.rand(2,32,48,4)
    result = node.EasyFaceBeautifyStatic.execute(image)
    assert torch.equal(result.args[0], image)
    assert not hasattr(result, 'ui')
    settings = json.loads(result.args[1])
    assert settings['source'] is None
    assert settings['colors'] == beauty.DEFAULT_BEAUTY_COLORS


def test_partial_preset_overrides_only_specified_numeric_controls(monkeypatch):
    node = load_node(monkeypatch)
    monkeypatch.setattr(beauty, 'detect_face', lambda _: [])
    monkeypatch.setattr(node, 'load_beauty_source', lambda _: pytest.fail('Numeric node must use IMAGE'))
    preset = json.dumps({'parameters': {'eyes': .8}, 'colors': {'lip':'#ff0000'},
                         'source_mode':'picker', 'source':{'source_type':'input','file_path':'unavailable.png'}})
    image = torch.rand(2,32,48,3)
    result = node.EasyFaceBeautifyStatic.execute(image, eyes=.2, smooth=.4, blush_color='#ff9f50', preset=preset)
    settings = json.loads(result.args[1])
    assert settings['parameters']['eyes'] == .8
    assert settings['parameters']['smooth'] == .4
    assert settings['colors']['lip'] == '#ff0000'
    assert settings['colors']['blush'] == '#ff9f50'
    assert settings['source'] is None
    assert 'source_mode' not in settings
    assert torch.equal(result.args[0],image)


def test_editor_preset_matches_numeric_output_on_entire_batch(monkeypatch):
    node = load_node(monkeypatch)
    monkeypatch.setattr(beauty, 'detect_face', lambda _: [[.5,.5]])
    monkeypatch.setattr(beauty, 'face_geometry', lambda *_: makeup_geometry())
    monkeypatch.setattr(node, 'beauty_preview', lambda *_: {'face_found':True})
    image = torch.full((2,64,64,3), .5)
    settings = beauty.parse_beauty_preset({'parameters':{'eye_bright':.6,'under_eye':.4,'lip_tint':.5,'blush':.2},
                                          'colors':{'lip':'#b8485f','blush':'#ed7e88'}})
    preset = json.dumps(settings)
    editor = node.EasyFaceBeautify.execute(preset, image)
    numeric = node.EasyFaceBeautifyStatic.execute(image, eyes=.9, preset=preset)
    assert torch.equal(editor.args[0],numeric.args[0])
    assert json.loads(numeric.args[1])['parameters']['eyes'] == 0
    assert not hasattr(numeric,'ui')


@pytest.mark.parametrize('preset', ['{', '[]', '{"parameters":{"eyes":"large"}}'])
def test_numeric_invalid_preset_fails_explicitly(monkeypatch,preset):
    node = load_node(monkeypatch)
    with pytest.raises(ValueError):
        node.EasyFaceBeautifyStatic.execute(torch.rand(1,32,32,3),preset=preset)


def test_static_hex_validation_and_empty_preset(monkeypatch):
    node = load_node(monkeypatch)
    with pytest.raises(ValueError,match='color'):
        node.EasyFaceBeautifyStatic.execute(torch.rand(1,32,32,3),lip_color='not-a-color')
    result = node.EasyFaceBeautifyStatic.execute(torch.rand(1,32,32,3),lip_color='#C4485F',blush_color='#AA8877',preset='  ')
    assert json.loads(result.args[1])['colors']['lip'] == '#c4485f'

    assert json.loads(result.args[1])['colors']['blush'] == '#aa8877'


def contour_landmarks():
    points = [[.5,.5] for _ in range(478)]
    values = {234:(.25,.45),454:(.75,.45),127:(.28,.25),356:(.72,.25),93:(.23,.42),323:(.77,.42),
              172:(.30,.65),397:(.70,.65),152:(.5,.83),148:(.42,.78),377:(.58,.78),10:(.5,.15),
              33:(.33,.38),133:(.44,.38),362:(.56,.38),263:(.67,.38),1:(.5,.5),61:(.42,.65),291:(.58,.65)}
    for index,point in values.items():
        points[index] = list(point)
    return points


@pytest.mark.parametrize('key', ['slim_face','small_face','narrow_face','temple_width','cheekbone_width','jaw_width','chin','pointed_chin'])
def test_each_face_control_warps_locally_and_reverses_direction(key):
    points = contour_landmarks()
    geo = beauty.face_geometry(points,1)
    y,x = torch.meshgrid(torch.linspace(0,1,128),torch.linspace(0,1,128),indexing='ij')
    image = torch.stack([x,y,(x+y)/2],-1)
    positive = beauty.parse_beauty_preset(json.dumps({'parameters':{key:.6}}))['parameters']
    negative = beauty.parse_beauty_preset(json.dumps({'parameters':{key:-.6}}))['parameters']
    out_positive = beauty.apply_face_beauty(image,positive,geo)
    out_negative = beauty.apply_face_beauty(image,negative,geo)
    assert (out_positive-image).abs().max() > .003
    assert (out_negative-image).abs().max() > .003
    assert torch.allclose(out_positive[0],image[0],atol=1e-6)
    # Opposite strengths produce opposite displacement on a linear coordinate image.
    forward = out_positive-image
    backward = out_negative-image
    assert (forward*backward).sum() < 0
    assert len(geo['handles']) <= beauty.MAX_BEAUTY_HANDLES


def test_detailed_face_defaults_and_static_preset_override(monkeypatch):
    node = load_node(monkeypatch)
    monkeypatch.setattr(beauty,'detect_face',lambda _: contour_landmarks())
    image = torch.rand(1,64,64,3)
    old = beauty.parse_beauty_preset('{"parameters":{"slim_face":0.2}}')
    assert all(old['parameters'][key]==0 for key in beauty.FACE_DETAIL_PARAMETERS)
    preset = json.dumps({'parameters':{'small_face':.4,'temple_width':-.2}})
    result = node.EasyFaceBeautifyStatic.execute(image,small_face=.1,jaw_width=.3,preset=preset)
    actual = json.loads(result.args[1])['parameters']
    assert actual['small_face']==.4 and actual['temple_width']==-.2 and actual['jaw_width']==.3
    assert torch.isfinite(result.args[0]).all()
