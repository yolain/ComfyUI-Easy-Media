declare module 'bun' {
  export const $: (
    strings: TemplateStringsArray,
    ...values: unknown[]
  ) => Promise<unknown>
}

declare const Bun: {
  argv: string[]
  build(options: unknown): Promise<{
    success: boolean
    logs: unknown[]
    outputs: unknown[]
  }>
}

interface ImportMeta {
  dir: string
  env: {
    PROJECT_NAME: string
  }
}
