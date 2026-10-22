// Everything personal that appears across the site lives here.

export const site = {
  name: 'Daniel Scalena',
  shortName: 'danielsc4',
  url: 'https://danielsc4.it',
  title: 'Daniel Scalena',
  description:
    'Daniel Scalena is a PhD researcher at the University of Milano-Bicocca and the University of Groningen, working on interpretability, steering and reasoning of large language models.',
  email: 'scalena99@gmail.com',
  cv: '/assets/pdf/Daniel_Scalena_CV_an.pdf',
  location: 'Europe',
  // Shown in the small panel on the home page. `logos` are files in public/logos/ (recoloured by CSS).
  status: [
    { label: 'Now', title: 'Member of Technical Staff, Intern', org: 'Cohere', logos: ['cohere'] },
    { label: 'PhD', title: 'Double PhD in NLP Interpretability', org: 'Milano-Bicocca & Groningen', logos: ['unimib', 'rug'] },
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
