import * as React from 'react'
import { Switch as SwitchPrimitive } from '@base-ui/react/switch'
import { cn } from '@/lib/utils'

const Switch = React.forwardRef<
  React.ComponentRef<typeof SwitchPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitive.Root
    ref={ref}
    className={(state) => cn(
      'inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border border-border p-0.5 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring',
      state.checked ? 'bg-primary' : 'bg-input',
      state.disabled && 'cursor-not-allowed opacity-50',
      typeof className === 'function' ? className(state) : className,
    )}
    {...props}
  >
    <SwitchPrimitive.Thumb
      className="pointer-events-none block size-4 shrink-0 rounded-full bg-background shadow-sm transition-transform"
      style={(state) => ({ transform: state.checked ? 'translateX(14px)' : 'translateX(0)' })}
    />
  </SwitchPrimitive.Root>
))
Switch.displayName = 'Switch'

export { Switch }
