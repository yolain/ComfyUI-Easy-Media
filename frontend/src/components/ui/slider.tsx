import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "@/lib/utils"

export interface SliderTick {
  value: number
  label: string
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> & {
    ticks?: SliderTick[]
  }
>(({ className, "aria-label": ariaLabel, "aria-labelledby": ariaLabelledBy, ticks, ...props }, ref) => {
  const value = props.value?.[0] ?? props.defaultValue?.[0]

  return (
    <div className={cn("relative", ticks && ticks.length > 0 && "pb-4")}>
      <SliderPrimitive.Root
        ref={ref}
        className={cn(
          "relative flex w-full touch-none select-none items-center",
          className
        )}
        aria-label={ariaLabel}
        aria-labelledby={ariaLabelledBy}
        {...props}
      >
        <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-muted">
          <SliderPrimitive.Range className="absolute h-full bg-primary" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          aria-label={ariaLabel}
          aria-labelledby={ariaLabelledBy}
          className="block h-3 w-3 rounded-full border border-primary/50 bg-primary shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
        />
        {ticks && ticks.length > 0 && (
          <div className="absolute left-0 top-full flex w-full justify-between">
            {ticks.map((tick) => {
              const isActive = tick.value === value
              return (
                <div key={tick.value} className="flex flex-col items-center gap-0.5">
                  <div
                    className={cn(
                      "h-1.5 w-px bg-muted-foreground/40",
                      isActive && "bg-primary"
                    )}
                  />
                  <span
                    className={cn(
                      "text-[10px] tabular-nums leading-none text-muted-foreground",
                      isActive && "font-medium text-foreground"
                    )}
                  >
                    {tick.label}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </SliderPrimitive.Root>
    </div>
  )
})
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }
