import { useId, useState } from 'react'
import { Check, ChevronDown, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useT } from '@/lib/i18n'

interface ProjectPickerProps {
  disabled: boolean
  value: string
  projects: string[]
  onOpen: () => void
  onSelect: (name: string) => void
}

export function ProjectPicker({ disabled, value, projects, onOpen, onSelect }: ProjectPickerProps) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const listId = useId()
  const matches = projects.filter((name) => name.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))

  return (
    <Popover open={open} onOpenChange={(nextOpen) => {
      setOpen(nextOpen)
      if (nextOpen) {
        setQuery('')
        onOpen()
      }
    }}>
      <PopoverTrigger asChild>
        <Button type="button" variant="outline" disabled={disabled} role="combobox" aria-expanded={open} aria-controls={listId} aria-label={t('projectVideoCombine.selectProject')} className="h-7 w-32 justify-between px-2 text-xs">
          <span className="min-w-0 truncate" title={value}>{value}</span>
          <ChevronDown className="size-3 shrink-0" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="flex max-h-[var(--radix-popover-content-available-height)] w-[200px] flex-col overflow-hidden p-1">
        <div className="flex shrink-0 items-center gap-2 rounded-md border border-input px-2 focus-within:ring-1 focus-within:ring-ring">
          <Search aria-hidden="true" className="pointer-events-none size-3.5 shrink-0 text-muted-foreground" />
          <Input className="h-8 min-w-0 flex-1 truncate border-0 px-0 text-xs shadow-none focus-visible:ring-0" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t('projectVideoCombine.searchProjects')} aria-label={t('projectVideoCombine.searchProjects')} />
        </div>
        <div id={listId} role="listbox" aria-label={t('projectVideoCombine.selectProject')} className="mt-1 h-60 min-h-0 overflow-y-auto overscroll-contain">
          {matches.map((name) => (
            <Button key={name} type="button" variant="ghost" role="option" aria-selected={name === value} className="h-8 w-full justify-start gap-2 px-2 text-xs" onClick={() => {
              setOpen(false)
              onSelect(name)
            }}>
              <Check className={`size-3 shrink-0 ${name === value ? '' : 'invisible'}`} />
              <span className="min-w-0 truncate" title={name}>{name}</span>
            </Button>
          ))}
          {matches.length === 0 && <div className="p-2 text-xs text-muted-foreground">{t('projectVideoCombine.noMatchingProjects')}</div>}
        </div>
      </PopoverContent>
    </Popover>
  )
}
