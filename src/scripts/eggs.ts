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

// Fake feature attribution over the page text, in the style of interpretability tools.
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
      node.textContent?.trim() && !node.parentElement?.closest('pre, button, script, style, summary')
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
  toast('<b>Attribution mode.</b> Token saliency for “who is Daniel?”. Scores are made up; the papers are not. Press I again to turn off.', 6000);
}

function togglePhoto(force?: boolean) {
  const on = force ?? !('photo' in root.dataset);
  if (on) {
    root.dataset.photo = '';
    toast('<b>Photo mode.</b> Move the pointer to relight the scene. P or Esc to exit.');
  } else {
    delete root.dataset.photo;
  }
}

const KONAMI = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
let konami = 0;

function breakaway() {
  window.dispatchEvent(new Event('terrain:boost'));
  toast('<b>Breakaway!</b> Out of the saddle, the valley flies by. Segment record unlocked.');
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
  const accent = getComputedStyle(root).getPropertyValue('--accent').trim() || '#4cb8ff';
  console.log(
    `%c
   ╭──────────╮
  ╭╯ ╭──────╮ ╰╮
  │  │  ●   │  │   danielsc4.it
  ╰╮ ╰──────╯ ╭╯
   ╰──────────╯
`,
    `color:${accent};font-family:monospace`,
  );
  console.log(
    '%cPeeking at the internals? Same, that is basically my research.\n\n' +
      '  I                 attribution mode\n' +
      '  P                 photo mode\n' +
      '  ↑↑↓↓←→←→ B A      breakaway\n' +
      '  5× the logo       early exit\n\n' +
      'Say hi: scalena99@gmail.com',
    'font-family:monospace;line-height:1.5',
  );
}

export function initEggs() {
  greet();
  document.querySelector('.brand')?.addEventListener('click', onBrandClick as EventListener);

  window.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as Element | null;
    if (target?.closest?.('input, textarea, select, [contenteditable]')) return;

    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    konami = key === KONAMI[konami] ? konami + 1 : key === KONAMI[0] ? 1 : 0;
    if (konami === KONAMI.length) {
      konami = 0;
      breakaway();
      return;
    }
    if (konami > 0) return;

    if (key === 'i') toggleAttribution();
    else if (key === 'p') togglePhoto();
    else if (key === 'Escape' && 'photo' in root.dataset) togglePhoto(false);
  });
}
