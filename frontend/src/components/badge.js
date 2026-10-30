export const badge = (c, t) => `<span class="badge ${c}">${t}</span>`;

if (typeof window !== 'undefined') {
  window.badge = badge;
}
