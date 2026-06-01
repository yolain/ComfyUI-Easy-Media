/**
 * Inject stylesheet contents into the document head with proper deduplication.
 */
export function addInlineStyles(css: string, id: string): HTMLStyleElement | null {
  if (document.getElementById(id)) {
    return null;
  }

  const style = document.createElement("style");
  style.id = id;
  style.textContent = css;
  document.head.appendChild(style);
  return style;
}
