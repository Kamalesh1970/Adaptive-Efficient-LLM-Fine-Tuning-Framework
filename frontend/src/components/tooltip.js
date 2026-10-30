// Tooltip for unfamiliar terms (hover, focus or touch)
export const T = (w, t) => `<span class="tip" tabindex="0" data-tip="${t}">${w}</span>`;
export const tooltip = T;

if (typeof window !== 'undefined') {
  window.T = T;
  window.tooltip = tooltip;
}
