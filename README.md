# danielsc4.it

Personal website, built with [Astro](https://astro.build) and deployed to GitHub Pages on every push to `main`.

## Everyday updates

| I want to…            | Edit                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| Add a publication     | Paste the BibTeX entry into `src/data/papers.bib`                    |
| Add a news item       | Create `src/content/news/YYYY-MM-DD-something.md`                    |
| Change the bio        | `src/data/about/academic.md` (and the casual/oversteered variants)  |
| Update position/PhD panel, links | `src/site.ts` (`status`, `socials`)                       |
| Update the CV         | Replace `public/assets/pdf/Daniel_Scalena_CV_an.pdf`                 |

### Publications

Entries are rendered newest first, grouped by `year` (within a year, in file order). The standard `url` field is the
"Paper" link: use the published version (ACL Anthology, journal, OpenReview) or arXiv for preprints. These optional
fields control the site and are stripped from the "Copy BibTeX" box:

```bibtex
  selected = {true},                 % show on the home page
  venue    = {ICML 2026},            % label; otherwise taken from "(...)" at the end of booktitle/journal/note
  website  = {https://...},          % project page
  pdf      = {paper.pdf},            % only if there is no url: file in public/assets/pdf/
  poster   = {poster.pdf},
  slides   = {slides.pdf},
  code     = {https://github.com/...},
  preview  = {image.png},            % file in src/assets/previews/ (shown for selected papers)
  abstract = {...},
```

Your name is highlighted automatically; mark equal contribution with `*` after the surname (`Daniel Scalena*`).

### News

```md
---
date: 2026-07-01
---

One or two sentences, Markdown links allowed.
```

The home page shows the latest four; `/news/` shows all of them.

### Easter eggs

All in `src/scripts/eggs.ts`, triggered by `data-egg` buttons in the text (styled to look like plain words):

- "look inside" in the hero lede (or `I`): fake token attribution over the page.
- "photography" in the bio (or `P`): hides the UI so the landscape is the picture.
- The α dial next to "About": swaps the bio between the three files in `src/data/about/`.
- Clicking the logo mark five times: early exit to the contact section.
- Dark theme between 20:00 and 06:00 shows a starry sky (`?night` forces it).
- A note for whoever reads the page source, in `src/layouts/Base.astro`.

## Local preview

```sh
npm install
npm run dev       # http://localhost:4321, reloads on save
npm run build     # production build into dist/
```

## Structure

- `src/pages/` home, publications, news and 404 pages
- `src/components/` header, footer, publication and news blocks
- `src/scripts/terrain.ts` the WebGL topographic background (no dependencies)
- `src/styles/global.css` all styles; theme colours are CSS variables at the top, the terrain reads them too
