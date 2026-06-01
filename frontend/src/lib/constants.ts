/**
 * Root CSS class applied to every React widget container.
 *
 * Derived at build time from `import.meta.env.PROJECT_NAME` (the project's
 * directory name, lowercased). Bun's `define` inlines the value into the bundle.
 *
 * This value is the single source of truth used by:
 *   - create-react-widget.ts  → container.classList.add(CUSTOM_NODE_CLASS)
 *   - tailwind.config.ts      → important: `.${CUSTOM_NODE_CLASS}`  (derives independently via path)
 *   - globals.css             → compiled, scoped, and embedded into index.js by build.ts
 */
export const CUSTOM_NODE_CLASS = (import.meta.env.PROJECT_NAME as string).toLowerCase()
