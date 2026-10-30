// DOM helper
export const $ = s => document.querySelector(s);

if (typeof window !== 'undefined') {
  window.$ = $;
}

