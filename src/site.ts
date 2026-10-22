// Everything personal that appears across the site lives here.

export const site = {
  name: 'Daniel Scalena',
  shortName: 'danielsc4',
  url: 'https://danielsc4.it',
  title: 'Daniel Scalena — Interpretability of Large Language Models',
  description:
    'Daniel Scalena is a PhD researcher at the University of Milano-Bicocca and the University of Groningen, working on interpretability, steering and reasoning of large language models.',
  tagline: 'I look inside language models to understand how they think, and use what I find to make them more reliable, controllable and capable.',
  email: 'scalena99@gmail.com',
  cv: '/assets/pdf/Daniel_Scalena_CV_an.pdf',
  location: 'Europe',
  // Most recent first. `logos` are files in public/logos/ (single-colour SVGs, recoloured by CSS).
  path: [
    {
      role: 'Member of Technical Staff, Intern',
      org: 'Cohere',
      detail: 'Code Agents team: reward design for code generation.',
      dates: 'Jun 2026 – Sep 2026',
      logos: ['cohere'],
    },
    {
      role: 'Double PhD in Computer Science, NLP',
      org: 'University of Milano-Bicocca & University of Groningen',
      detail: 'Interpretability for safer, more reliable and more useful generative models.',
      dates: 'Nov 2023 – 2026',
      logos: ['unimib', 'rug'],
    },
    {
      role: 'M.Sc. in Computer Science, ML track',
      org: 'University of Milano-Bicocca',
      detail: '110/110 cum laude. Thesis on the explainability of LLM detoxification.',
      dates: 'Oct 2021 – Oct 2023',
      logos: ['unimib'],
    },
  ],
  affiliations: [
    { name: 'University of Milano-Bicocca', url: 'https://en.unimib.it/' },
    { name: 'University of Groningen', url: 'https://rug.nl/' },
  ],
  socials: [
    { label: 'Scholar', url: 'https://scholar.google.com/citations?user=5Q89Yd8AAAAJ' },
    { label: 'GitHub', url: 'https://github.com/danielsc4' },
    { label: 'Bluesky', url: 'https://bsky.app/profile/danielsc4.it' },
    { label: 'X', url: 'https://x.com/daniel_sc4' },
    { label: 'LinkedIn', url: 'https://www.linkedin.com/in/daniel-scalena' },
    { label: 'ORCID', url: 'https://orcid.org/0009-0006-0518-6504' },
    { label: 'Semantic Scholar', url: 'https://www.semanticscholar.org/author/2221318599' },
  ],
} as const;
