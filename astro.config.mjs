import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://danielsc4.it',
  integrations: [sitemap()],
  redirects: {
    '/cv': '/assets/pdf/Daniel_Scalena_CV_an.pdf',
  },
  build: {
    inlineStylesheets: 'always',
  },
  devToolbar: {
    enabled: false,
  },
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'hover',
  },
});
