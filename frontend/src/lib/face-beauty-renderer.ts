import { BEAUTY_PARAMETERS, DEFAULT_BEAUTY_COLORS, type BeautyGeometry, type BeautyParameters } from '@/lib/face-beauty'

const VERTEX = `#version 300 es
in vec2 position;
out vec2 uv;
void main() { uv = (position + 1.) * .5; gl_Position = vec4(position.x, -position.y, 0., 1.); }`
const FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform sampler2D featureMask;
uniform vec4 accents[4];
uniform float eye_bright;
uniform float under_eye;
uniform float lip_tint;
uniform float blush;
uniform vec3 lipColor;
uniform vec3 blushColor;
uniform float aspect;
uniform int count;
uniform vec4 handles[7];
uniform vec3 changes[7];
uniform vec4 regions[4];
uniform vec2 axis;
uniform float blurRadius;
uniform float smoothing;
uniform float whiten;
uniform float brightness;
uniform float warmth;
in vec2 uv;
out vec4 color;
vec4 sampleAt(vec2 p) { return texture(image, clamp(p / vec2(1., aspect), vec2(0.), vec2(1.))); }
float luma(vec3 c) { return dot(c, vec3(.299,.587,.114)); }
float accentAt(vec2 p, vec4 region) {
  vec2 d = p - region.xy;
  vec2 local = vec2(dot(d, axis), dot(d, vec2(-axis.y, axis.x)));
  return 1. - smoothstep(.45, 1., length(local / max(region.zw, vec2(.00001))));
}
void main() {
  vec2 p = uv * vec2(1., aspect);
  for (int i=0; i<7; i++) {
    if (i >= count) break;
    vec2 d = p - handles[i].xy;
    float t = clamp(length(d) / handles[i].z, 0., 1.);
    float fall = (1. - t*t*(3. - 2.*t)) * handles[i].w;
    p = handles[i].xy + d * (1. - changes[i].z * fall) - changes[i].xy * fall;
  }
  vec4 src = sampleAt(p);
  float skin = count > 0 ? 1. : 0.;
  for (int i=0; i<4; i++) {
    vec2 d = p - regions[i].xy;
    vec2 local = vec2(dot(d, axis), dot(d, vec2(-axis.y, axis.x)));
    float m = 1. - smoothstep(.8, 1., length(local / max(regions[i].zw, vec2(.00001))));
    skin *= i == 0 ? m : 1. - m;
  }
  vec3 col = src.rgb;
  if (smoothing > 0. && count > 0) {
    vec3 blurred = vec3(0.);
    for (int x=-1; x<=1; x++) for (int y=-1; y<=1; y++) blurred += sampleAt(p + vec2(float(x), float(y)) * blurRadius).rgb / 9.;
    float edge = smoothstep(.030, .115, abs(dot(col - blurred, vec3(.299,.587,.114))));
    col = mix(col, blurred, smoothing * skin * (1. - edge));
  }
  col += (1. - col) * skin * whiten * .25;
  col += skin * brightness * .15;
  col += skin * warmth * .06 * vec3(1., .15, -1.);
  if (count > 0) {
    if (under_eye > 0.) {
      float mask = max(accentAt(p, accents[0]), accentAt(p, accents[1]));
      float amount = mask * under_eye * (1. - smoothstep(.55, .85, luma(col)));
      vec3 lifted = col + (1. - col) * .16;
      lifted = mix(lifted, vec3(luma(lifted)), .12);
      col = mix(col, lifted, amount);
    }
    vec2 features = texture(featureMask, clamp(p / vec2(1., aspect), vec2(0.), vec2(1.))).rg;
    if (eye_bright > 0.) {
      float light = luma(col);
      float amount = features.g * eye_bright * smoothstep(.25, .65, light);
      vec3 target = mix(col, vec3(light), .65);
      target += (1. - target) * .18;
      col = mix(col, target, amount);
    }
    if (lip_tint > 0.) {
      vec3 target = lipColor * (.45 + .65 * luma(col));
      col = mix(col, target, features.r * lip_tint * .65);
    }
    if (blush > 0.) {
      vec3 base = clamp(col, 0., 1.);
      vec3 low = (1. - 2. * blushColor) * base * base + 2. * blushColor * base;
      vec3 high = 2. * base * (1. - blushColor) + sqrt(base) * (2. * blushColor - 1.);
      vec3 target = mix(low, high, step(vec3(.500001), blushColor));
      float amount = max(accentAt(p, accents[2]), accentAt(p, accents[3])) * blush * .6 * skin;
      col = mix(col, target, amount);
    }
  }
  color = vec4(clamp(col, 0., 1.), src.a);
}`

/** RGB channels carry lip and eye/teeth coverage, independent of display colors. */
function createFeatureMask(width: number, height: number, geometry: BeautyGeometry): Uint8Array {
  function paint(polygons: [number, number][][], cutout?: [number, number][]) {
    const canvas = document.createElement('canvas')
    canvas.width = width; canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Unable to prepare face feature masks')
    const polygon = (points: [number, number][]) => {
      if (points.length < 3) return
      context.beginPath()
      points.forEach(([x,y], index) => { if (index === 0) context.moveTo(Math.round(x * width), Math.round(y * width)); else context.lineTo(Math.round(x * width), Math.round(y * width)) })
      context.closePath(); context.fill()
    }
    polygons.forEach(polygon)
    if (cutout) { context.globalCompositeOperation = 'destination-out'; polygon(cutout) }
    return context.getImageData(0,0,width,height).data
  }
  const features = geometry.features
  const lips = paint(features ? [features.lip_outer] : [], features?.lip_inner)
  const eyes = paint(features ? [...features.eyes, features.lip_inner] : [])
  const data = new Uint8Array(width * height * 4)
  for (let i=0; i<data.length; i+=4) { data[i] = lips[i+3]; data[i+1] = eyes[i+3]; data[i+3] = 255 }
  return data
}

/** GPU preview; the Python renderer uses the same metric, handles and sampling. */
export function createBeautyRenderer(canvas: HTMLCanvasElement, source: HTMLImageElement, geometry: BeautyGeometry) {
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, alpha: false })
  if (!gl) throw new Error('Face Beauty requires WebGL2')
  canvas.width = source.naturalWidth
  canvas.height = source.naturalHeight
  const shaders: WebGLShader[] = []
  let program: WebGLProgram | null = null
  let buffer: WebGLBuffer | null = null
  let texture: WebGLTexture | null = null
  let featureTexture: WebGLTexture | null = null
  const dispose = () => {
    if (texture) gl.deleteTexture(texture)
    if (featureTexture) gl.deleteTexture(featureTexture)
    if (buffer) gl.deleteBuffer(buffer)
    if (program) gl.deleteProgram(program)
    shaders.forEach(shader => gl.deleteShader(shader))
  }
  try {
    program = gl.createProgram()
    if (!program) throw new Error('Unable to create face beauty shader program')
    for (const [type, code] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, FRAGMENT]] as const) {
      const shader = gl.createShader(type)
      if (!shader) throw new Error('Unable to create face beauty shader')
      shaders.push(shader)
      gl.shaderSource(shader, code)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? 'Shader compilation failed')
      gl.attachShader(program, shader)
    }
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'Shader link failed')
    gl.useProgram(program)
    buffer = gl.createBuffer()
    texture = gl.createTexture()
    if (!buffer || !texture) throw new Error('Unable to allocate face beauty GPU resources')
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW)
    const location = gl.getAttribLocation(program, 'position')
    gl.enableVertexAttribArray(location)
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source)
    featureTexture = gl.createTexture()
    if (!featureTexture) throw new Error('Unable to allocate face feature mask')
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, featureTexture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, createFeatureMask(canvas.width, canvas.height, geometry))
    gl.activeTexture(gl.TEXTURE0)
    const uniform = (name: string) => gl.getUniformLocation(program!, name)
    gl.uniform1i(uniform('featureMask'), 1)
    const accents = new Float32Array(16)
    geometry.accents?.forEach((r, i) => accents.set([...r.center, ...r.radii], i * 4))
    gl.uniform4fv(uniform('accents[0]'), accents)
    gl.uniform1f(uniform('aspect'), canvas.height / canvas.width)
    gl.uniform1i(uniform('image'), 0)
    gl.uniform1i(uniform('count'), geometry.handles.length)
    const regions = new Float32Array(16)
    geometry.regions.forEach((r, i) => regions.set([...r.center, ...r.radii], i * 4))
    gl.uniform4fv(uniform('regions[0]'), regions)
    gl.uniform2fv(uniform('axis'), geometry.axis ?? [1, 0])
    gl.uniform1f(uniform('blurRadius'), geometry.blur ?? 0)
    const changes = new Float32Array(21)
    geometry.handles.forEach((h, i) => changes.set([...h.delta, h.scale], i * 3))
    gl.uniform3fv(uniform('changes[0]'), changes)
    return {
      draw(parameters: BeautyParameters, colors = DEFAULT_BEAUTY_COLORS) {
        if (gl.isContextLost()) throw new Error('Face Beauty GPU context was lost; reload the image')
        gl.useProgram(program)
        const handles = new Float32Array(28)
        geometry.handles.forEach((h, i) => handles.set([...h.center, h.radius, parameters[h.key]], i * 4))
        gl.uniform4fv(uniform('handles[0]'), handles)
        for (const key of BEAUTY_PARAMETERS) {
          if (key === 'smooth') gl.uniform1f(uniform('smoothing'), parameters[key])
          else if (['whiten', 'brightness', 'warmth', 'eye_bright', 'under_eye', 'lip_tint', 'blush'].includes(key)) gl.uniform1f(uniform(key), parameters[key])
        }
        for (const key of ['lip', 'blush'] as const) {
          const hex = colors[key]
          gl.uniform3fv(uniform(key === 'lip' ? 'lipColor' : 'blushColor'), [1,3,5].map(index => parseInt(hex.slice(index, index+2),16) / 255))
        }
        gl.viewport(0, 0, canvas.width, canvas.height)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      },
      dispose,
    }
  } catch (error) {
    dispose()
    throw error
  }
}
