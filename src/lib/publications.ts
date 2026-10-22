import { parse, type Creator } from '@retorquere/bibtex-parser';
import type { ImageMetadata } from 'astro';
import bibSource from '../data/papers.bib?raw';

// Fields that only drive the website and are stripped from the copyable BibTeX.
const SITE_FIELDS = new Set(['abstract', 'selected', 'preview', 'pdf', 'poster', 'website', 'code', 'slides', 'venue']);

const ME = 'scalena';

const previews = import.meta.glob<{ default: ImageMetadata }>('../assets/previews/*', { eager: true });

export interface Author {
  name: string;
  me: boolean;
  equal: boolean;
}

export interface PubLink {
  label: string;
  url: string;
}

export interface Publication {
  key: string;
  title: string;
  authors: Author[];
  year: number;
  venue: string;
  selected: boolean;
  abstract?: string;
  preview?: ImageMetadata;
  links: PubLink[];
  bibtex: string;
}

function asset(value: string): string {
  return /^https?:\/\//.test(value) ? value : `/assets/pdf/${value}`;
}

function toAuthor(c: Creator): Author {
  if ((c.name ?? c.lastName) === 'others') return { name: 'et al.', me: false, equal: false };
  if (c.name) return { name: c.name, me: false, equal: false };
  const last = (c.lastName ?? '').trim();
  const equal = last.endsWith('*');
  const clean = last.replace(/\*$/, '');
  const name = [c.firstName, c.prefix, clean].filter(Boolean).join(' ');
  return { name, me: clean.toLowerCase() === ME, equal };
}

function venueOf(f: Record<string, any>, type: string): string {
  if (f.venue) return f.venue;
  const source: string | undefined = f.booktitle || f.journal || f.note;
  if (source) {
    const short = source.match(/\(([^()]+)\)\s*$/);
    return short ? short[1].trim() : source;
  }
  if (type === 'misc' && (f.eprint || f.archiveprefix)) return 'Preprint';
  return '';
}

// Removes website-only fields from a raw BibTeX entry, keeping the original formatting of the rest.
function cleanBibtex(raw: string): string {
  const open = raw.indexOf('{');
  const headerEnd = raw.indexOf(',', open);
  const header = raw.slice(0, headerEnd + 1).trim();
  const body = raw.slice(headerEnd + 1, raw.lastIndexOf('}'));

  const fields: string[] = [];
  let depth = 0;
  let inQuote = false;
  let current = '';
  for (const ch of body) {
    if (ch === '{') depth++;
    else if (ch === '}') depth--;
    else if (ch === '"' && depth === 0) inQuote = !inQuote;
    if (ch === ',' && depth === 0 && !inQuote) {
      fields.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current);

  const kept = fields
    .map((field) => field.trim())
    .filter(Boolean)
    .filter((field) => !SITE_FIELDS.has(field.split('=')[0].trim().toLowerCase()))
    .map((field) => '  ' + field.replace(/\s*\n\s*/g, ' '));

  return `${header}\n${kept.join(',\n')}\n}`;
}

function build(): Publication[] {
  const library = parse(bibSource, { english: false, caseProtection: false, unsupported: 'text' });
  if (library.errors.length) {
    throw new Error(`papers.bib has errors: ${JSON.stringify(library.errors)}`);
  }

  const pubs = library.entries.map((entry): Publication => {
    const f = entry.fields as Record<string, any>;
    const links: PubLink[] = [];
    const paper = f.url || (f.pdf ? asset(f.pdf) : undefined);
    if (paper) links.push({ label: 'Paper', url: paper });
    if (f.website) links.push({ label: 'Website', url: f.website });
    if (f.code) links.push({ label: 'Code', url: f.code });
    if (f.poster) links.push({ label: 'Poster', url: asset(f.poster) });
    if (f.slides) links.push({ label: 'Slides', url: asset(f.slides) });

    const previewFile = f.preview ? previews[`../assets/previews/${f.preview}`] : undefined;
    if (f.preview && !previewFile) {
      throw new Error(`Preview "${f.preview}" for ${entry.key} not found in src/assets/previews/`);
    }

    return {
      key: entry.key,
      title: f.title ?? entry.key,
      authors: (f.author ?? []).map(toAuthor),
      year: Number(f.year) || 0,
      venue: venueOf(f, entry.type),
      selected: String(f.selected).toLowerCase() === 'true',
      abstract: f.abstract,
      preview: previewFile?.default,
      links,
      bibtex: cleanBibtex(entry.input),
    };
  });

  return pubs.sort((a, b) => b.year - a.year);
}

export const publications = build();
export const selectedPublications = publications.filter((p) => p.selected);

export function groupByYear(list: Publication[]): [number, Publication[]][] {
  const groups = new Map<number, Publication[]>();
  for (const pub of list) {
    groups.set(pub.year, [...(groups.get(pub.year) ?? []), pub]);
  }
  return [...groups.entries()];
}
