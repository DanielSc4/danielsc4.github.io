// Small, optional surprises. Nothing here is needed for the site to work.

const root = document.documentElement;

let toastEl: HTMLDivElement | null = null;
let toastTimer = 0;

function toast(html: string, ms = 4200) {
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.className = 'egg-toast mono';
    toastEl.setAttribute('role', 'status');
    document.body.append(toastEl);
  }
  toastEl.innerHTML = html;
  requestAnimationFrame(() => toastEl!.classList.add('is-visible'));
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toastEl!.classList.remove('is-visible'), ms);
}

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return ((h >>> 0) % 10000) / 10000;
}

// "Look inside": fake feature attribution over the page text, in the style of interpretability tools.
const SALIENT =
  /^(interpretab|steer|model|language|reason|llm|safe|reliab|control|capab|agent|generat|entropy|chain|translat|design|cit(y|ies)|travel|photo|explor)/i;
let attribution: Map<HTMLElement, Text> | null = null;

function toggleAttribution() {
  finishFlap?.();
  if (attribution) {
    attribution.forEach((original, wrapper) => wrapper.replaceWith(original));
    attribution = null;
    toast('Attribution off', 2000);
    return;
  }
  attribution = new Map();
  const main = document.querySelector('main');
  if (!main) return;
  const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.textContent?.trim() && !node.parentElement?.closest('pre, button, script, style, summary, template, label')
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT,
  });
  const nodes: Text[] = [];
  while (walker.nextNode()) nodes.push(walker.currentNode as Text);

  let index = 0;
  for (const node of nodes) {
    const wrapper = document.createElement('span');
    for (const part of node.textContent!.split(/(\s+)/)) {
      if (!part || /^\s+$/.test(part)) {
        wrapper.append(part);
        continue;
      }
      const r = hash(part + index++);
      const score = SALIENT.test(part.replace(/^\W+/, '')) ? 0.7 + r * 0.3 : Math.pow(r, 3) * 0.55;
      const token = document.createElement('span');
      token.className = score > 0.88 ? 'tok hot' : 'tok';
      token.style.setProperty('--s', score.toFixed(2));
      token.textContent = part;
      wrapper.append(token);
    }
    node.replaceWith(wrapper);
    attribution.set(wrapper, node);
  }
  toast('<b>Attribution mode</b>', 2500);
}

// "Photography": hide the interface and let the landscape be the picture.
function exitPhotoOnClick() {
  togglePhoto(false);
}

function togglePhoto(force?: boolean) {
  const on = force ?? !('photo' in root.dataset);
  if (on) {
    root.dataset.photo = '';
    toast('<b>Photo mode</b>', 2500);
    setTimeout(() => document.addEventListener('click', exitPhotoOnClick, { once: true }), 0);
  } else {
    delete root.dataset.photo;
    document.removeEventListener('click', exitPhotoOnClick);
  }
}

// Split-flap transition, like an old station departures board: every letter of the new text
// flips through the alphabet, starting from the letter that was there before.
const FLAPS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?'’-()&α";
const FLAP_MS = 38;
let finishFlap: (() => void) | null = null;

function flapSteps(from: string, to: string): string[] {
  const max = 4 + Math.floor(Math.random() * 9);
  const a = FLAPS.indexOf(from);
  const b = FLAPS.indexOf(to);
  if (a < 0 || b < 0) {
    return [...Array.from({ length: max }, () => FLAPS[1 + Math.floor(Math.random() * 52)]), to];
  }
  const distance = (b - a + FLAPS.length) % FLAPS.length;
  const start = distance > max ? (b - max + FLAPS.length) % FLAPS.length : a;
  const steps = [FLAPS[start]];
  for (let i = start; i !== b; ) {
    i = (i + 1) % FLAPS.length;
    steps.push(FLAPS[i]);
  }
  return steps;
}

function splitFlap(target: HTMLElement, next: Node) {
  finishFlap?.();
  const before = (target.textContent ?? '').replace(/\s+/g, ' ').trim();
  target.replaceChildren(next);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.textContent?.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT),
  });
  const nodes: Text[] = [];
  while (walker.nextNode()) nodes.push(walker.currentNode as Text);

  type Cell = { el: HTMLSpanElement; steps: string[]; delay: number; shown: number };
  const cells: Cell[] = [];
  const restore: [HTMLElement, Text][] = [];
  let index = 0;
  for (const node of nodes) {
    const wrapper = document.createElement('span');
    for (const part of node.data.split(/(\s+)/)) {
      if (!part) continue;
      if (/^\s+$/.test(part)) {
        wrapper.append(part);
        index++;
        continue;
      }
      const word = document.createElement('span');
      word.className = 'flap-word';
      for (const ch of part) {
        const el = document.createElement('span');
        el.className = 'flap';
        el.textContent = ch;
        word.append(el);
        const delay = Math.min(index * 1.4, 650) + Math.random() * 90;
        cells.push({ el, steps: flapSteps(before[index] ?? ' ', ch), delay, shown: -1 });
        index++;
      }
      wrapper.append(word);
    }
    node.replaceWith(wrapper);
    restore.push([wrapper, node]);
  }

  // Lock every cell to the width of its final letter so the text doesn't reflow while flipping.
  const widths = cells.map((c) => c.el.getBoundingClientRect().width);
  cells.forEach((c, i) => {
    c.el.style.width = `${widths[i]}px`;
    c.el.textContent = c.steps[0];
  });

  let raf = 0;
  const t0 = performance.now();
  finishFlap = () => {
    cancelAnimationFrame(raf);
    restore.forEach(([wrapper, node]) => wrapper.replaceWith(node));
    finishFlap = null;
  };
  const tick = (now: number) => {
    let done = true;
    for (const c of cells) {
      const step = Math.min(c.steps.length - 1, Math.max(0, Math.floor((now - t0 - c.delay) / FLAP_MS)));
      if (step < c.steps.length - 1) done = false;
      if (step === c.shown) continue;
      c.shown = step;
      c.el.textContent = c.steps[step];
      if (step > 0) c.el.className = c.el.className === 'flap fa' ? 'flap fb' : 'flap fa';
    }
    if (done) finishFlap?.();
    else raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

// The α dial next to "About": steer the bio's tone, and over-steer it until fluency collapses.
const STRENGTH = ['0', '4', '12'];

function initSteering() {
  const input = document.querySelector<HTMLInputElement>('[data-steer]');
  const output = document.querySelector<HTMLOutputElement>('[data-steer-value]');
  const target = document.querySelector<HTMLElement>('[data-steer-target]');
  if (!input || !output || !target) return;

  input.addEventListener('input', () => {
    const level = input.value;
    output.value = STRENGTH[Number(level)];
    const template = document.querySelector<HTMLTemplateElement>(`template[data-steer-level="${level}"]`);
    if (!template) return;
    if (attribution) toggleAttribution();
    splitFlap(target, template.content.cloneNode(true));
  });
}

// The logo mark: on the home page it scrolls to the top; five quick clicks exit early.
let clicks: number[] = [];
function onBrandClick(event: MouseEvent) {
  if (location.pathname !== '/') return;
  event.preventDefault();
  const now = performance.now();
  clicks = [...clicks.filter((t) => now - t < 1800), now];
  if (clicks.length < 5) {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }
  clicks = [];
  toast('<b>Commitment boundary reached.</b> Exiting early…', 3000);
  setTimeout(() => document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' }), 700);
}

function greet() {
  console.info(
    '%c◉ danielsc4.it%c\nPeeking at the internals? Same, that is basically my research.\nSome words on this page do more than they say. Say hi: scalena99@gmail.com',
    'font:600 14px monospace;color:#4cb8ff',
    'font:12px/1.6 monospace',
  );
}

export function initEggs() {
  greet();
  initSteering();
  document.querySelector('.brand')?.addEventListener('click', onBrandClick as EventListener);

  document.addEventListener('click', (event) => {
    const trigger = (event.target as Element | null)?.closest?.<HTMLElement>('[data-egg]');
    if (!trigger) return;
    event.stopPropagation();
    if (trigger.dataset.egg === 'inspect') toggleAttribution();
    else if (trigger.dataset.egg === 'photo') togglePhoto(true);
  });

  window.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as Element | null;
    if (target?.closest?.('input, textarea, select, [contenteditable]')) return;
    const key = event.key.toLowerCase();
    if (key === 'i') toggleAttribution();
    else if (key === 'p') togglePhoto();
    else if (key === 'escape' && 'photo' in root.dataset) togglePhoto(false);
  });
}
