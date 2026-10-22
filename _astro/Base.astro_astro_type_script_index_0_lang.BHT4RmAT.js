var e={uBg:`--bg`,uSurface:`--terrain-surface`,uLine:`--terrain-line`,uAccent:`--accent`,uAccent2:`--accent-2`,uGlow:`--terrain-glow`};function t(e){let t=e.trim().replace(`#`,``),n=t.length===3?[...t].map(e=>e+e).join(``):t,r=parseInt(n,16);return[(r>>16&255)/255,(r>>8&255)/255,(r&255)/255]}function n(e,t,n,r){let i=1/Math.tan(e/2),a=1/(n-r);return new Float32Array([i/t,0,0,0,0,i,0,0,0,0,(r+n)*a,-1,0,0,2*r*n*a,0])}function r(e,t){let n=e[0]-t[0],r=e[1]-t[1],i=e[2]-t[2],a=Math.hypot(n,r,i);n/=a,r/=a,i/=a;let o=i,s=-n,c=Math.hypot(o,s);o/=c,s/=c;let l=r*s,u=i*o-n*s,d=-r*o;return new Float32Array([o,l,n,0,0,u,r,0,s,d,i,0,-(o*e[0]+s*e[2]),-(l*e[0]+u*e[1]+d*e[2]),-(n*e[0]+r*e[1]+i*e[2]),1])}function i(e,t){let n=new Float32Array(16);for(let r=0;r<4;r++)for(let i=0;i<4;i++)n[r*4+i]=e[i]*t[r*4]+e[4+i]*t[r*4+1]+e[8+i]*t[r*4+2]+e[12+i]*t[r*4+3];return n}function a(e){return-Math.sin(e*.21)*1.6-Math.sin(e*.07+1.3)*1.2}function o(e,t,n){let r=e.createShader(t);if(e.shaderSource(r,n),e.compileShader(r),!e.getShaderParameter(r,e.COMPILE_STATUS))throw Error(e.getShaderInfoLog(r)??`shader compile failed`);return r}function s(e,t,n){let r=e.createProgram();return e.attachShader(r,o(e,e.VERTEX_SHADER,t)),e.attachShader(r,o(e,e.FRAGMENT_SHADER,n)),e.linkProgram(r),e.getProgramParameter(r,e.LINK_STATUS)?r:(console.warn(e.getProgramInfoLog(r)),null)}function c(){if(new URLSearchParams(location.search).has(`night`))return!0;let e=new Date().getHours();return e>=22||e<6}function l(o){let l=o.getContext(`webgl2`,{antialias:!0,alpha:!1,powerPreference:`low-power`});if(!l)return;let u=s(l,`#version 300 es
precision highp float;
in vec2 aGrid;
uniform mat4 uViewProj;
uniform vec3 uCam;
uniform vec2 uOffset;   // lattice-snapped x offset and travel
uniform vec2 uRemainder; // sub-cell remainder, keeps motion smooth without vertex swimming
uniform vec2 uCell;

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
`,`#version 300 es
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
`),d=s(l,`#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`,`#version 300 es
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
`);if(!u||!d)return;let f=l.createVertexArray(),p=e=>l.getUniformLocation(d,e),m={res:p(`uRes`),shift:p(`uShift`),time:p(`uTime`),fade:p(`uFade`),bg:p(`uBg`),star:p(`uStar`),tint:p(`uTint`)};l.useProgram(u);let h=matchMedia(`(pointer: coarse)`).matches,g=h?150:210,_=h?190:250,v=52/(g-1),y=37.5/(_-1),b=new Float32Array(g*_*2);for(let e=0;e<_;e++)for(let t=0;t<g;t++){let n=(e*g+t)*2;b[n]=-26+t*v,b[n+1]=1.5-e*y}let x=new Uint16Array((g-1)*(_-1)*6),S=0;for(let e=0;e<_-1;e++)for(let t=0;t<g-1;t++){let n=e*g+t,r=n+g;x.set([n,r,n+1,n+1,r,r+1],S),S+=6}let C=l.createVertexArray();l.bindVertexArray(C),l.bindBuffer(l.ARRAY_BUFFER,l.createBuffer()),l.bufferData(l.ARRAY_BUFFER,b,l.STATIC_DRAW);let w=l.getAttribLocation(u,`aGrid`);l.enableVertexAttribArray(w),l.vertexAttribPointer(w,2,l.FLOAT,!1,0,0),l.bindBuffer(l.ELEMENT_ARRAY_BUFFER,l.createBuffer()),l.bufferData(l.ELEMENT_ARRAY_BUFFER,x,l.STATIC_DRAW),l.enable(l.DEPTH_TEST);let T=e=>l.getUniformLocation(u,e),E={viewProj:T(`uViewProj`),cam:T(`uCam`),offset:T(`uOffset`),remainder:T(`uRemainder`),cell:T(`uCell`),light:T(`uLight`),dark:T(`uDark`),time:T(`uTime`),fade:T(`uFade`),far:T(`uFar`)};l.uniform2f(E.cell,v,y),l.uniform1f(E.far,34);let D=document.documentElement,O=o.dataset.intensity===`subtle`,k=!1;function A(){let n=getComputedStyle(D);for(let[r,i]of Object.entries(e))l.uniform3fv(T(r),t(n.getPropertyValue(i)));let r=t(n.getPropertyValue(`--bg`));l.clearColor(...r,1);let i=D.dataset.theme===`dark`;l.uniform1f(E.dark,+!!i),l.uniform1f(T(`uStrength`),parseFloat(n.getPropertyValue(`--terrain-strength`))||1),k=i&&c(),l.useProgram(d),l.uniform3fv(m.bg,r),l.uniform3fv(m.star,t(n.getPropertyValue(`--ink`))),l.uniform3fv(m.tint,t(n.getPropertyValue(`--accent`))),l.useProgram(u)}A(),setInterval(A,6e4),new MutationObserver(()=>{A(),U()}).observe(D,{attributes:!0,attributeFilter:[`data-theme`,`data-photo`]});let j=Math.min(window.devicePixelRatio||1,h?1.25:1.5);function M(){let e=Math.round(o.clientWidth*j),t=Math.round(o.clientHeight*j),n=o.width!==e||o.height!==t;return n&&(o.width=e,o.height=t),l.viewport(0,0,o.width,o.height),n}new ResizeObserver(()=>{if(!M())return U();cancelAnimationFrame(R),H(performance.now())}).observe(o),M();let N=matchMedia(`(prefers-reduced-motion: reduce)`),P={x:0,y:0,tx:0,ty:0};window.addEventListener(`pointermove`,e=>{P.tx=e.clientX/innerWidth*2-1,P.ty=e.clientY/innerHeight*2-1,U()},{passive:!0}),window.addEventListener(`scroll`,()=>U(),{passive:!0});let F=performance.now(),I=scrollY*.012,L=0,R=0,z=F,B=0,V=0;function H(e){R=0;let t=N.matches,s=t?0:(e-F)/1e3,c=e-z;z=e,t||(L+=Math.min(c,50)/1e3*.22);let p=`photo`in D.dataset;V++>10&&V<90&&c>24&&++B>20&&j>.75&&(j=Math.max(.75,j*.7),B=-1e3,M());let h=p?0:scrollY/innerHeight,g=scrollY*.012+L,_=t?1:.08;I+=(g-I)*_,P.x+=(P.tx-P.x)*(t?1:.05),P.y+=(P.ty-P.y)*(t?1:.05);let b=-I,S=a(b)*.85+P.x*.45,w=1.35-P.y*.18,T=O&&!p,A=Math.min(1,h)*(T?.6:1),H=o.width/o.height,W=H<1?.5:0,G=[a(b-7)*.85+P.x*.2,1.15+W-A*3-(T?1.4:0),-7],K=[S,w,0],q=H<1?.95+(1-H)*.35:.86;l.uniformMatrix4fv(E.viewProj,!1,i(n(q,H,.05,60),r(K,G))),l.uniform3f(E.cam,...K);let J=Math.floor(I/y)*y,Y=Math.floor(S/v)*v;l.uniform2f(E.offset,Y,J),l.uniform2f(E.remainder,0,I-J),l.uniform3f(E.light,-.5+P.x*.9,.45-P.y*.25,-1),l.uniform1f(E.time,s);let X=T?.42:1-.8*Math.min(1,Math.max(0,h/.9));l.uniform1f(E.fade,X),l.clear(l.COLOR_BUFFER_BIT|l.DEPTH_BUFFER_BIT),k&&(l.useProgram(d),l.bindVertexArray(f),l.disable(l.DEPTH_TEST),l.uniform2f(m.res,o.width,o.height),l.uniform2f(m.shift,P.x*.02,-A*.45),l.uniform1f(m.time,s),l.uniform1f(m.fade,X),l.drawArrays(l.TRIANGLES,0,3),l.enable(l.DEPTH_TEST),l.bindVertexArray(C),l.useProgram(u)),l.drawElements(l.TRIANGLES,x.length,l.UNSIGNED_SHORT,0),o.classList.contains(`is-ready`)||o.classList.add(`is-ready`),!t&&!document.hidden&&U()}function U(){R||=requestAnimationFrame(H)}document.addEventListener(`visibilitychange`,()=>{document.hidden||U()}),N.addEventListener(`change`,U),o.addEventListener(`webglcontextlost`,()=>cancelAnimationFrame(R)),U()}var u=(function(){let e=typeof document<`u`&&document.createElement(`link`).relList;return e&&e.supports&&e.supports(`modulepreload`)?`modulepreload`:`preload`})(),d=function(e){return`/`+e},f={},p=function(e){return e.pathname.endsWith(`.css`)},m=function(e,t,n){if(t in e)return e[t];let r=n();if(!r){e[t]=void 0;return}let i=r.then(()=>{e[t]=void 0},n=>{throw e[t]=void 0,n});return e[t]=i,i},h=function(e,t,n){let r=Promise.resolve();if(t&&t.length>0){let e,i=document.querySelector(`meta[property=csp-nonce]`),a=i?.nonce||i?.getAttribute(`nonce`);function o(e){return Promise.all(e.map(e=>Promise.resolve(e).then(e=>({status:`fulfilled`,value:e}),e=>({status:`rejected`,reason:e}))))}function s(e){return import.meta.resolve?new URL(import.meta.resolve(e)):new URL(e,import.meta.url)}r=o(t.map(t=>{t=d(t,n);let r=s(t),i=p(r);return m(f,r.href,()=>{if(e===void 0){e={all:new Set,styles:new Set};let t=document.getElementsByTagName(`link`);for(let n=t.length-1;n>=0;n--){let r=t[n];e.all.add(r.href),r.rel===`stylesheet`&&e.styles.add(r.href)}}if((i?e.styles:e.all).has(r.href))return;let t=document.createElement(`link`);if(t.rel=i?`stylesheet`:u,i||(t.as=`script`),t.crossOrigin=``,t.href=r.href,a&&t.setAttribute(`nonce`,a),document.head.appendChild(t),i)return new Promise((e,n)=>{t.addEventListener(`load`,e),t.addEventListener(`error`,()=>n(Error(`Unable to preload CSS for ${r}`)))})})}).filter(e=>e!==void 0))}function i(e){let t=new Event(`vite:preloadError`,{cancelable:!0});if(t.payload=e,window.dispatchEvent(t),!t.defaultPrevented)throw e}return r.then(t=>{for(let e of t||[])e.status===`rejected`&&i(e.reason);return e().catch(i)})},g=document.getElementById(`terrain`),_=()=>{try{l(g)}catch(e){console.warn(`Terrain disabled:`,e)}h(()=>import(`./eggs.BcIqdnPS.js`).then(e=>e.initEggs()),[])};`requestIdleCallback`in window?requestIdleCallback(_,{timeout:800}):setTimeout(_,200);