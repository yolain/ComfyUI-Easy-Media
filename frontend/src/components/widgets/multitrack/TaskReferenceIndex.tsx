import { cn } from '@/lib/utils'

interface TaskReferenceIndexProps {
  index: number
  kind?: 'image' | 'mod'
  testId: string
}

export function TaskReferenceIndex({ index, kind = 'image', testId }: Readonly<TaskReferenceIndexProps>) {
  return (
    <span
      data-testid={testId}
      className={cn(
        'absolute bottom-0 left-0 z-10 min-w-5 rounded-sm px-1.5 py-0.5 text-center text-[9px] font-semibold leading-none',
        kind === 'mod' ? 'bg-refmod text-refmod-foreground' : 'bg-black/50 text-white',
      )}
    >
      {kind === 'mod' ? `mod${index}` : index}
    </span>
  )
}
