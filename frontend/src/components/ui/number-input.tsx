"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { ChevronUp, ChevronDown } from "lucide-react"

interface NumberInputProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  className?: string
  disabled?: boolean
  commitOnBlur?: boolean
}

export function NumberInput({
  value,
  onChange,
  min = 0,
  max = Infinity,
  step = 1,
  className,
  disabled,
  commitOnBlur = false,
}: NumberInputProps) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const blurOnEnterRef = React.useRef(false)
  const [draftValue, setDraftValue] = React.useState<string>(String(value))

  // Sync draftValue when value prop changes (for parent updates)
  React.useEffect(() => {
    if (!commitOnBlur) return
    setDraftValue(String(value))
  }, [value, commitOnBlur])

  const clamp = React.useCallback(
    (val: number): number => {
      return Math.min(max, Math.max(min, val))
    },
    [min, max]
  )

  const parseAndCommit = React.useCallback(
    (rawValue: string) => {
      if (rawValue === "") {
        onChange(clamp(0))
        return
      }
      const parsed = Number.parseFloat(rawValue)
      if (!Number.isNaN(parsed)) {
        onChange(clamp(parsed))
      }
    },
    [onChange, clamp]
  )

  const handleIncrement = React.useCallback(() => {
    // Buttons always commit immediately
    const newValue = Math.min(max, value + step)
    onChange(newValue)
    if (commitOnBlur) {
      setDraftValue(String(newValue))
    }
  }, [commitOnBlur, value, step, max, onChange])

  const handleDecrement = React.useCallback(() => {
    // Buttons always commit immediately
    const newValue = Math.max(min, value - step)
    onChange(newValue)
    if (commitOnBlur) {
      setDraftValue(String(newValue))
    }
  }, [commitOnBlur, value, step, min, onChange])

  const handleInputChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const rawValue = e.currentTarget.value
      if (commitOnBlur) {
        setDraftValue(rawValue)
      } else {
        if (rawValue === "") {
          onChange(0)
          return
        }
        const parsed = Number.parseFloat(rawValue)
        if (!Number.isNaN(parsed)) {
          onChange(clamp(parsed))
        }
      }
    },
    [commitOnBlur, onChange, clamp]
  )

  const handleBlur = React.useCallback(() => {
    if (commitOnBlur && !blurOnEnterRef.current) {
      parseAndCommit(draftValue)
      setDraftValue(String(value))
    }
    blurOnEnterRef.current = false
  }, [commitOnBlur, parseAndCommit, draftValue, value])

  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "ArrowUp") {
        e.preventDefault()
        // Arrow keys always commit immediately
        const newValue = Math.min(max, value + step)
        onChange(newValue)
        if (commitOnBlur) {
          setDraftValue(String(newValue))
        }
      } else if (e.key === "ArrowDown") {
        e.preventDefault()
        // Arrow keys always commit immediately
        const newValue = Math.max(min, value - step)
        onChange(newValue)
        if (commitOnBlur) {
          setDraftValue(String(newValue))
        }
      } else if (commitOnBlur && e.key === "Enter") {
        e.preventDefault()
        blurOnEnterRef.current = true
        parseAndCommit(draftValue)
        inputRef.current?.blur()
      }
    },
    [commitOnBlur, value, step, min, max, onChange, parseAndCommit, draftValue]
  )

  return (
    <div
      className={cn(
        "flex h-7 items-center rounded-md border border-node-input-border bg-node-input overflow-hidden",
        className
      )}
    >
      <input
        type="number"
        value={commitOnBlur ? draftValue : value}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        ref={inputRef}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={cn(
          "w-14 h-full flex-1 bg-muted text-node-foreground text-xs text-center border-0 focus:outline-none focus:ring-0 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        )}
      />
      <div className="flex flex-col h-full border-l ">
        <button
          type="button"
          onClick={handleIncrement}
          disabled={disabled || value >= max}
          aria-label="Increment"
          className={cn(
            "flex-1 flex items-center justify-center px-1 hover:bg-card cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed bg-muted"
          )}
        >
          <ChevronUp className="size-3 text-node-foreground" />
        </button>
        <div className="h-px bg-border" />
        <button
          type="button"
          onClick={handleDecrement}
          disabled={disabled || value <= min}
          aria-label="Decrement"
          className={cn(
            "flex-1 flex items-center justify-center px-1 hover:bg-card cursor-pointer transition-colors disabled:opacity-40 disabled:cursor-not-allowed bg-muted"
          )}
        >
          <ChevronDown className="size-3 text-node-foreground" />
        </button>
      </div>
    </div>
  )
}
