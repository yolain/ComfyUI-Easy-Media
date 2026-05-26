import { Plus } from 'lucide-react'
import { useT } from '@/lib/i18n'

interface InsertButtonProps {
  position: 'left' | 'right'
  onClick: () => void
  disabled?: boolean
}

export function InsertButton({ position, onClick, disabled }: Readonly<InsertButtonProps>) {
  const t = useT()
  return (
    <button
      type="button"
      aria-label={position === 'left' ? t('insertButton.insertBefore') : t('insertButton.insertAfter')}
      className="flex items-center opacity-80 hover:opacity-100 transition-opacity cursor-pointer select-none disabled:opacity-20"
      onClick={(e) => { e.stopPropagation(); onClick() }}
      disabled={disabled}
      style={{ padding: 0, border: 'none', background: 'none' }}
    >
      <span
        className="inline-flex items-center justify-center rounded-full bg-foreground shrink-0"
        style={{ width: 10, height: 10 }}
      >
        <Plus className="w-2 h-2 text-background" />
      </span>
    </button>
  )
}