# ComfyUI Easy Media

## Tech Stack

- **Backend**: Python, using comfyui-nodes-** skills to develop.
- **Frontend**: React v19, shadcn/ui + Tailwind CSS v4 for components, Lucide for icons, Bun for package manager and build, Vitest for testing.

---

## Build, Test, and Development Commands

### Adding shadcn/ui components

```bash
# From repo root
bunx shadcn add <component-name>
```

## Build

```bash
# Dev build (unminified, outputs to dist/dev) — use during normal development
bun run dev               # watch mode with hot reload → dist/dev
bun run build             # one-shot dev build → dist/dev

# Release build (minified, outputs to dist/release) — REQUIRED before git commit
bun run build:release     # production build → dist/release
```

## Code Style

### Backend

- **Reuse utils first** — Before writing new code, check `utils/` for existing utility functions; if none exists, add it there instead of duplicating code
- **File naming** — `kebab_case.py` for nodes、utils、modules; `PascalCase.py` for classes
- **Type hints** — Use type hints for all public methods and function parameters
- **Error handling** — Use `try/except` to explicitly catch exceptions; never swallow exceptions silently


### Frontend

- **TypeScript first** — strict mode, no `any` without reason
- **File naming** — `PascalCase.tsx` for components, `camelCase` for dir, `kebab-case.ts` for modules
- **Import aliases** — use `@/*` for `src/` paths
- **React components** — functional only, hooks-based
- **Reusable hooks** — When component logic has reusable state/effect behavior, write it directly in `src/hooks` and call it from components instead of duplicating the logic inline
- **Test location** — Write test cases separately under `src/tests`; do not colocate test files with components or other source modules
- **Never use raw HTML elements** for interactive controls — use shadcn/ui equivalents (`<Button>` not `<button>`, `<Input>` not `<input>`, `<Textarea>` not `<textarea>`, `<Select>` not `<select>`). Exception: `src/components/ui/**` (the shadcn/ui primitives themselves may use raw elements)
- **Error handling** — always handle errors explicitly; never swallow exceptions silently
- **Color tokens only** — All colors are defined as CSS custom properties in `src/styles/global.css` and mapped to Tailwind utilities via `@theme inline`; use theme tokens instead of hardcoded values
- **No raw Tailwind colors** — Do not use raw color utilities such as `text-zinc-*`, `border-slate-*`, `bg-[#...]`, or `text-[#...]`; use classes like `bg-background`, `border-border`, `text-foreground`, and `text-muted-foreground`
- **Color exceptions** — `bg-white` and `bg-black` are allowed for canvas/node editor backgrounds when explicit backgrounds are intentional
- **Adding colors** — Add new colors as CSS custom property tokens in `src/styles/index.css` under `@theme inline`, then consume them through Tailwind utilities

---

## Git Workflow

### Branch Strategy

- Switch to `main` and ensure it is up to date before creating a new feature branch when the task requires branch work
- If a matching feature branch already exists, use it instead of creating a duplicate
- For code changes, create feature branches from `main`; documentation-only or assistant-rules changes do not require a new branch unless requested
- When fixing code from an issue, create a branch specifically for that issue
- Before switching or committing, check whether the current branch is behind `main` with conflicting changes; rebase `main` into the current branch if needed

### Commit Rules

- Group commits by feature or responsibility, not only by file type
- Keep `dist/` changes separate from feature/source commits when frontend builds update generated output
- Use `git diff` to inspect changes and decide logical commit grouping before committing
- Use conventional commit messages:

```text
<type>: <description>

<optional body>
```

- Allowed commit types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

### Pull Request Workflow

- After pushing a branch to remote, create a corresponding PR when requested or when completing branch-based work
- If a similar PR is already open, ask whether to contribute to that PR instead
- When creating PRs, analyze the full branch history, not only the latest commit
- Use `git diff [base-branch]...HEAD` to review the full change set
- Include a concise PR summary and a test plan
- Push with `-u` when publishing a new branch

### Pre-Push Review

- Run applicable reviews before pushing substantial code changes:
  - Frontend changes: React/TypeScript review
  - Backend changes: Python/ComfyUI node review
  - Localization changes: i18n review
- Address review findings and re-run review after major changes
