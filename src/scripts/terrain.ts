// Procedural topographic terrain rendered with raw WebGL2.
// The camera follows a winding "route" through a valley; scrolling flies forward,
// the pointer steers the camera and the light. Colours come from CSS custom properties.

const NOISE = /* glsl */ `
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}
float gnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(dot(hash2(i), f), dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
    mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)), dot(hash2(i + vec2(1.0)), f - vec2(1.0)), u.x),
    u.y);
}
const mat2 ROT = mat2(1.6, 1.2, -1.2, 1.6);
float fbm(vec2 p) {
  float h = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) { h += a * gnoise(p); p = ROT * p; a *= 0.5; }
  return h;
}
float ridged(vec2 p) {
  float h = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) {
    float n = clamp(1.0 - abs(gnoise(p) * 2.0), 0.0, 1.0);
    h += a * n * n;
    p = ROT * p + vec2(5.2, 1.3);
    a *= 0.5;
  }
  return h;
}
float route(float z) {
  return -sin(z * 0.21) * 1.6 - sin(z * 0.07 + 1.3) * 1.2;
}
float height(vec2 s) {
  vec2 p = s * 0.23;
  float h = fbm(p) * 0.9 + pow(ridged(p * 0.8 + 3.1), 2.0) * 1.35;
  float v = smoothstep(0.6, 4.0, abs(s.x - route(s.y)));
  return (h * (0.2 + 0.8 * v) + v * 0.45 - 0.3) * 1.1;
}
`;

const VERT = /* glsl */ `#version 300 es
precision highp float;
in vec2 aGrid;
uniform mat4 uViewProj;
uniform vec3 uCam;
uniform vec2 uOffset;   // lattice-snapped x offset and travel
uniform vec2 uRemainder; // sub-cell remainder, keeps motion smooth without vertex swimming
uniform vec2 uCell;
${NOISE}
out vec3 vPos;
out vec3 vN;
out float vH;
out vec2 vS;
out float vDist;
void main() {
  vec3 world = vec3(aGrid.x + uOffset.x, 0.0, aGrid.y + uRemainder.y);
  vec2 s = vec2(world.x, aGrid.y - uOffset.y);
  float h = height(s);
  float hx = height(s + vec2(uCell.x, 0.0));
  float hz = height(s + vec2(0.0, uCell.y));
  vN = normalize(vec3((h - hx) / uCell.x, 1.0, (h - hz) / uCell.y));
  world.y = h;
  vPos = world;
  vH = h;
  vS = s;
  vDist = length(world - uCam);
  gl_Position = uViewProj * vec4(world, 1.0);
}
`;

const FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec3 vPos;
in vec3 vN;
in float vH;
in vec2 vS;
in float vDist;
uniform vec3 uCam;
uniform vec3 uLight;
uniform vec3 uBg;
uniform vec3 uSurface;
uniform vec3 uLine;
uniform vec3 uAccent;
uniform vec3 uAccent2;
uniform vec3 uGlow;
uniform float uDark;
uniform float uTime;
uniform float uFade;
uniform float uStrength;
uniform float uFar;
out vec4 outColor;
float route(float z) {
  return -sin(z * 0.21) * 1.6 - sin(z * 0.07 + 1.3) * 1.2;
}
float isoline(float v, float lo, float hi) {
  float w = max(fwidth(v), 1e-4);
  float d = abs(fract(v + 0.5) - 0.5) / w;
  return (1.0 - smoothstep(lo, hi, d)) * (1.0 - smoothstep(0.3, 0.8, w));
}
void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(uCam - vPos);
  vec3 L = normalize(uLight);
  vec3 H = normalize(L + V);
  float diff = clamp(dot(N, L) * 0.5 + 0.5, 0.0, 1.0);
  float ndv = clamp(dot(N, V), 0.0, 1.0);
  float spec = pow(max(dot(N, H), 0.0), 60.0);
  float fres = pow(1.0 - ndv, 4.0);

  // Thin-film style iridescence on reflections, restricted to the theme palette.
  float k = abs(fract(ndv * 1.3 + vH * 0.2 + uTime * 0.02) * 2.0 - 1.0);
  vec3 irid = mix(mix(uAccent, uGlow, smoothstep(0.0, 0.5, k)), uAccent2, smoothstep(0.5, 1.0, k));
  vec3 col = uSurface * mix(0.8, 1.06, diff);
  col += (spec * 0.55 + fres * 0.12) * mix(vec3(1.0), irid, 0.75) * mix(0.3, 1.0, uDark);

  // Contour lines: minor every step, major every fifth.
  float c = vH * 8.0;
  float lines = max(isoline(c, 0.4, 1.4) * 0.5, isoline(c / 5.0, 0.6, 2.0));

  // A scanning pulse sweeping towards the horizon.
  float bandZ = 4.0 - mod(uTime * 5.0, 52.0);
  float scan = exp(-pow((vPos.z - bandZ) * 0.5, 2.0));
  vec3 lineCol = mix(uLine, uAccent2, scan * 0.85);
  col = mix(col, lineCol, clamp(lines * mix(0.5, 0.85, uDark) * (1.0 + scan * 0.7), 0.0, 1.0));

  // The route: a dashed track along the valley floor.
  float rd = abs(vS.x - route(vS.y));
  float dash = smoothstep(0.55, 0.6, fract(vS.y * 1.4));
  float rw = max(fwidth(rd), 1e-4);
  float track = (1.0 - smoothstep(0.012, 0.012 + rw * 1.5, rd)) * dash * smoothstep(1.5, 4.5, vDist);
  col = mix(col, uAccent, track * 0.85);
  col += uAccent * exp(-rd * rd * 5.0) * 0.22 * uDark;

  // Faint survey grid.
  vec2 g = vS * 0.5;
  vec2 gw = max(fwidth(g), vec2(1e-4));
  vec2 gd = abs(fract(g + 0.5) - 0.5) / gw;
  float grid = (1.0 - smoothstep(0.5, 1.5, min(gd.x, gd.y))) * (1.0 - smoothstep(0.15, 0.5, max(gw.x, gw.y)));
  col = mix(col, uLine, grid * mix(0.05, 0.1, uDark));

  // Fog into the page background, with a soft glow near the horizon.
  float f = clamp(pow(vDist / uFar, 1.6), 0.0, 1.0);
  float horizon = smoothstep(0.5, 0.9, f) * (1.0 - smoothstep(0.9, 1.0, f));
  vec3 fogCol = mix(uBg, uGlow, horizon * mix(0.12, 0.25, uDark));
  col = mix(col, fogCol, f);
  col = mix(uBg, col, uFade * uStrength);
  outColor = vec4(col, 1.0);
}
`;

// Night sky, drawn behind the terrain only in the dark theme at night (or with ?night).
const SKY_VERT = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

const SKY_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform vec2 uShift;
uniform float uTime;
uniform float uFade;
uniform vec3 uBg;
uniform vec3 uStar;
uniform vec3 uTint;
out vec4 outColor;
float h21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float stars(vec2 uv, float cellPx, float density, float seed) {
  vec2 g = uv * uRes.y / cellPx;
  vec2 id = floor(g);
  float r = h21(id + seed);
  if (r < density) return 0.0;
  vec2 jitter = vec2(h21(id + seed + 3.1), h21(id + seed + 7.7)) - 0.5;
  float d = length(fract(g) - 0.5 - jitter * 0.6) * cellPx;
  float size = mix(1.0, 2.2, (r - density) / (1.0 - density)) * uRes.y / 900.0;
  float twinkle = 0.55 + 0.45 * sin(uTime * (0.8 + r * 2.5) + r * 60.0);
  return (1.0 - smoothstep(0.0, size, d)) * twinkle;
}
void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y + uShift;
  float cell = uRes.y / 900.0;
  float s = stars(uv, 10.0 * cell, 0.975, 0.0) * 0.6 + stars(uv, 26.0 * cell, 0.93, 11.0);
  s *= smoothstep(-0.15, 0.35, uv.y);

  // An occasional shooting star.
  float period = 11.0;
  float slot = floor(uTime / period);
  float p = fract(uTime / period) * period;
  vec2 from = vec2(h21(vec2(slot, 1.0)) * 1.4 - 0.7, 0.25 + h21(vec2(slot, 2.0)) * 0.25);
  vec2 dir = normalize(vec2(h21(vec2(slot, 3.0)) > 0.5 ? 1.0 : -1.0, -0.45));
  float life = clamp(p / 0.9, 0.0, 1.0);
  vec2 head = from + dir * life * 0.6;
  vec2 tail = head - dir * 0.18 * life;
  vec2 pa = uv - tail, ba = head - tail;
  float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
  float d = length(pa - ba * t) * 900.0;
  float meteor = (1.0 - smoothstep(0.0, 1.6, d)) * t * step(p, 0.9) * sin(life * 3.14159);

  vec3 col = uBg + uStar * s * 0.85 + mix(uStar, uTint, 0.4) * meteor;
  outColor = vec4(mix(uBg, col, uFade), 1.0);
}
`;

type Vec3 = [number, number, number];

const COLOR_VARS = {
  uBg: '--bg',
  uSurface: '--terrain-surface',
  uLine: '--terrain-line',
  uAccent: '--accent',
  uAccent2: '--accent-2',
  uGlow: '--terrain-glow',
} as const;

function hexToRgb(hex: string): Vec3 {
  const h = hex.trim().replace('#', '');
  const full = h.length === 3 ? [...h].map((c) => c + c).join('') : h;
  const n = parseInt(full, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

function lookAt(eye: Vec3, target: Vec3): Float32Array {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  const zl = Math.hypot(zx, zy, zz);
  zx /= zl;
  zy /= zl;
  zz /= zl;
  // x = normalize(cross(up, z)) with up = (0, 1, 0)
  let xx = zz;
  let xz = -zx;
  const xl = Math.hypot(xx, xz);
  xx /= xl;
  xz /= xl;
  const yx = zy * xz;
  const yy = zz * xx - zx * xz;
  const yz = -zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    0, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

function multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function route(z: number): number {
  return -Math.sin(z * 0.21) * 1.6 - Math.sin(z * 0.07 + 1.3) * 1.2;
}

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? 'shader compile failed');
  }
  return shader;
}

function link(gl: WebGL2RenderingContext, vert: string, frag: string): WebGLProgram | null {
  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vert));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.warn(gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

function isNight(): boolean {
  if (new URLSearchParams(location.search).has('night')) return true;
  const hour = new Date().getHours();
  return hour >= 22 || hour < 6;
}

export function startTerrain(canvas: HTMLCanvasElement): void {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'low-power' });
  if (!gl) return;

  const program = link(gl, VERT, FRAG);
  const sky = link(gl, SKY_VERT, SKY_FRAG);
  if (!program || !sky) return;
  const skyVao = gl.createVertexArray();
  const su = (name: string) => gl.getUniformLocation(sky, name);
  const skyUniforms = {
    res: su('uRes'),
    shift: su('uShift'),
    time: su('uTime'),
    fade: su('uFade'),
    bg: su('uBg'),
    star: su('uStar'),
    tint: su('uTint'),
  };
  gl.useProgram(program);

  const coarse = matchMedia('(pointer: coarse)').matches;
  const cols = coarse ? 150 : 210;
  const rows = coarse ? 190 : 250;
  const halfWidth = 26;
  const near = 1.5;
  const far = -36;
  const dx = (halfWidth * 2) / (cols - 1);
  const dz = (near - far) / (rows - 1);

  const verts = new Float32Array(cols * rows * 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = (r * cols + c) * 2;
      verts[i] = -halfWidth + c * dx;
      verts[i + 1] = near - r * dz;
    }
  }
  const indices = new Uint16Array((cols - 1) * (rows - 1) * 6);
  let k = 0;
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c;
      const b = a + cols;
      indices.set([a, b, a + 1, a + 1, b, b + 1], k);
      k += 6;
    }
  }

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, 'aGrid');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  gl.enable(gl.DEPTH_TEST);

  const u = (name: string) => gl.getUniformLocation(program, name);
  const uniforms = {
    viewProj: u('uViewProj'),
    cam: u('uCam'),
    offset: u('uOffset'),
    remainder: u('uRemainder'),
    cell: u('uCell'),
    light: u('uLight'),
    dark: u('uDark'),
    time: u('uTime'),
    fade: u('uFade'),
    far: u('uFar'),
  };
  gl.uniform2f(uniforms.cell, dx, dz);
  gl.uniform1f(uniforms.far, Math.abs(far) - 2);

  const root = document.documentElement;
  const subtle = canvas.dataset.intensity === 'subtle';

  let night = false;
  function applyTheme() {
    const styles = getComputedStyle(root);
    for (const [uniform, cssVar] of Object.entries(COLOR_VARS)) {
      gl!.uniform3fv(u(uniform), hexToRgb(styles.getPropertyValue(cssVar)));
    }
    const bg = hexToRgb(styles.getPropertyValue('--bg'));
    gl!.clearColor(...bg, 1);
    const dark = root.dataset.theme === 'dark';
    gl!.uniform1f(uniforms.dark, dark ? 1 : 0);
    gl!.uniform1f(u('uStrength'), parseFloat(styles.getPropertyValue('--terrain-strength')) || 1);

    night = dark && isNight();
    gl!.useProgram(sky);
    gl!.uniform3fv(skyUniforms.bg, bg);
    gl!.uniform3fv(skyUniforms.star, hexToRgb(styles.getPropertyValue('--ink')));
    gl!.uniform3fv(skyUniforms.tint, hexToRgb(styles.getPropertyValue('--accent')));
    gl!.useProgram(program);
  }
  applyTheme();
  setInterval(applyTheme, 60_000);
  new MutationObserver(() => {
    applyTheme();
    requestFrame();
  }).observe(root, { attributes: true, attributeFilter: ['data-theme', 'data-photo'] });

  let dpr = Math.min(window.devicePixelRatio || 1, coarse ? 1.25 : 1.5);
  function resize() {
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl!.viewport(0, 0, canvas.width, canvas.height);
  }
  new ResizeObserver(() => {
    resize();
    requestFrame();
  }).observe(canvas);
  resize();

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  window.addEventListener(
    'pointermove',
    (e) => {
      pointer.tx = (e.clientX / innerWidth) * 2 - 1;
      pointer.ty = (e.clientY / innerHeight) * 2 - 1;
      requestFrame();
    },
    { passive: true },
  );
  window.addEventListener('scroll', () => requestFrame(), { passive: true });

  const start = performance.now();
  let travel = scrollY * 0.012;
  let drift = 0;
  let raf = 0;
  let lastFrame = start;
  let slowFrames = 0;
  let frames = 0;

  function render(now: number) {
    raf = 0;
    const still = reducedMotion.matches;
    const t = still ? 0 : (now - start) / 1000;
    const delta = now - lastFrame;
    lastFrame = now;
    if (!still) drift += (Math.min(delta, 50) / 1000) * 0.22;
    const photo = 'photo' in root.dataset;

    // Lower the resolution once if the device struggles.
    if (frames++ > 10 && frames < 90 && delta > 24 && ++slowFrames > 20 && dpr > 0.75) {
      dpr = Math.max(0.75, dpr * 0.7);
      slowFrames = -1000;
      resize();
    }

    const progress = photo ? 0 : scrollY / innerHeight;
    const targetTravel = scrollY * 0.012 + drift;
    const ease = still ? 1 : 0.08;
    travel += (targetTravel - travel) * ease;
    pointer.x += (pointer.tx - pointer.x) * (still ? 1 : 0.05);
    pointer.y += (pointer.ty - pointer.y) * (still ? 1 : 0.05);

    const camZs = -travel;
    const camX = route(camZs) * 0.85 + pointer.x * 0.45;
    const camY = 1.35 - pointer.y * 0.18;
    const calm = subtle && !photo;
    const pitch = Math.min(1, progress) * (calm ? 0.6 : 1);
    const aspect = canvas.width / canvas.height;
    const lookDist = 7;
    const portraitLift = aspect < 1 ? 0.5 : 0;
    const target: Vec3 = [
      route(camZs - lookDist) * 0.85 + pointer.x * 0.2,
      1.15 + portraitLift - pitch * 3.0 - (calm ? 1.4 : 0),
      -lookDist,
    ];
    const eye: Vec3 = [camX, camY, 0];

    const fov = aspect < 1 ? 0.95 + (1 - aspect) * 0.35 : 0.86;
    gl!.uniformMatrix4fv(uniforms.viewProj, false, multiply(perspective(fov, aspect, 0.05, 60), lookAt(eye, target)));
    gl!.uniform3f(uniforms.cam, ...eye);

    const snappedTravel = Math.floor(travel / dz) * dz;
    const snappedX = Math.floor(camX / dx) * dx;
    gl!.uniform2f(uniforms.offset, snappedX, snappedTravel);
    gl!.uniform2f(uniforms.remainder, 0, travel - snappedTravel);
    gl!.uniform3f(uniforms.light, -0.5 + pointer.x * 0.9, 0.45 - pointer.y * 0.25, -1);
    gl!.uniform1f(uniforms.time, t);
    const fade = calm ? 0.42 : 1 - 0.8 * Math.min(1, Math.max(0, progress / 0.9));
    gl!.uniform1f(uniforms.fade, fade);

    gl!.clear(gl!.COLOR_BUFFER_BIT | gl!.DEPTH_BUFFER_BIT);
    if (night) {
      gl!.useProgram(sky);
      gl!.bindVertexArray(skyVao);
      gl!.disable(gl!.DEPTH_TEST);
      gl!.uniform2f(skyUniforms.res, canvas.width, canvas.height);
      gl!.uniform2f(skyUniforms.shift, pointer.x * 0.02, -pitch * 0.45);
      gl!.uniform1f(skyUniforms.time, t);
      gl!.uniform1f(skyUniforms.fade, fade);
      gl!.drawArrays(gl!.TRIANGLES, 0, 3);
      gl!.enable(gl!.DEPTH_TEST);
      gl!.bindVertexArray(vao);
      gl!.useProgram(program);
    }
    gl!.drawElements(gl!.TRIANGLES, indices.length, gl!.UNSIGNED_SHORT, 0);

    if (!canvas.classList.contains('is-ready')) canvas.classList.add('is-ready');

    if (!still && !document.hidden) requestFrame();
  }

  function requestFrame() {
    if (!raf) raf = requestAnimationFrame(render);
  }

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) requestFrame();
  });
  reducedMotion.addEventListener('change', requestFrame);
  canvas.addEventListener('webglcontextlost', () => cancelAnimationFrame(raf));

  requestFrame();
}
