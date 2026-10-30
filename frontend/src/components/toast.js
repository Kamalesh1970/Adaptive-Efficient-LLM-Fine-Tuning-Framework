export function toast(m) {
  const t = document.querySelector("#toast");
  if (!t) return;
  t.textContent = m;
  t.classList.add("on");
  clearTimeout(toast.k);
  toast.k = setTimeout(() => t.classList.remove("on"), 2200);
}

if (typeof window !== 'undefined') {
  window.toast = toast;
}
