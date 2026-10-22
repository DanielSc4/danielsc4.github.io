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
  if (attribution) {
    attribution.forEach((original, wrapper) => wrapper.replaceWith(original));
    attribution = null;
    toast('Attribution off. Back to plain text.');
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
  toast('<b>Attribution mode.</b> Token saliency for “who is Daniel?”. Scores are made up; the papers are not. Look inside again to switch off.', 6000);
}

// "Photography": hide the interface and let the landscape be the picture.
function exitPhotoOnClick() {
  togglePhoto(false);
}

function togglePhoto(force?: boolean) {
  const on = force ?? !('photo' in root.dataset);
  if (on) {
    root.dataset.photo = '';
    toast('<b>Photo mode.</b> Move the pointer to relight the scene. Click anywhere to come back.');
    setTimeout(() => document.addEventListener('click', exitPhotoOnClick, { once: true }), 0);
  } else {
    delete root.dataset.photo;
    document.removeEventListener('click', exitPhotoOnClick);
  }
}

// The α dial next to "About": steer the bio's tone, and over-steer it until fluency collapses.
const STRENGTH = ['0', '4', '12'];

function initSteering() {
  const input = document.querySelector<HTMLInputElement>('[data-steer]');
  const output = document.querySelector<HTMLOutputElement>('[data-steer-value]');
  const target = document.querySelector<HTMLElement>('[data-steer-target]');
  if (!input || !output || !target) return;

  let pending = 0;
  input.addEventListener('input', () => {
    const level = input.value;
    output.value = STRENGTH[Number(level)];
    const template = document.querySelector<HTMLTemplateElement>(`template[data-steer-level="${level}"]`);
    if (!template) return;
    if (attribution) toggleAttribution();
    target.classList.add('is-steering');
    clearTimeout(pending);
    pending = window.setTimeout(() => {
      target.replaceChildren(template.content.cloneNode(true));
      target.classList.remove('is-steering');
    }, 220);
  });
}

let clicks: number[] = [];
function onBrandClick(event: MouseEvent) {
  const now = performance.now();
  clicks = [...clicks.filter((t) => now - t < 1800), now];
  if (clicks.length < 5) return;
  event.preventDefault();
  clicks = [];
  toast('<b>Commitment boundary reached.</b> The answer is already decided; the rest of this page is epiphenomenal. Exiting early…', 5000);
  setTimeout(() => document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' }), 900);
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
