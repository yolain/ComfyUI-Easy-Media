import { createContext, useContext } from 'react'
import en from '../../messages/en.json'
import zh from '../../messages/zh.json'

const messages = { en, zh } as const
type Locale = keyof typeof messages
type MessageSection = Record<string, string>

export const LocaleContext = createContext<string | undefined>(undefined)

function resolveLocale(raw: string | undefined): Locale {
  if (!raw) return 'en'
  const lang = raw.split('-')[0].toLowerCase()
  return (lang in messages ? lang : 'en') as Locale
}

export function translate(
  raw: string | undefined,
  path: string,
  params?: Record<string, string | number>,
): string {
  const locale = resolveLocale(raw)
  const m = messages[locale] as Record<string, MessageSection>
  const dot = path.indexOf('.')
  const section = dot === -1 ? path : path.slice(0, dot)
  const key = dot === -1 ? '' : path.slice(dot + 1)
  let str = m[section]?.[key] ?? path
  if (params) {
    for (const [name, value] of Object.entries(params)) {
      str = str.replaceAll(`{${name}}`, String(value))
    }
  }
  return str
}

/**
 * Returns a `t(path, params?)` function that resolves a dot-path key against
 * the current locale's message catalog.
 *
 * @example
 * const t = useT()
 * t('toolbar.frames')                          // "Frames" | "帧"
 * t('promptTrack.segmentLabel', { n: 2 })      // "Segment 2" | "片段 2"
 */
export function useT() {
  const raw = useContext(LocaleContext)
  return (path: string, params?: Record<string, string | number>) => translate(raw, path, params)
}
