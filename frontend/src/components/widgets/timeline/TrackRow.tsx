import { useState } from 'react'
import { MessageSquareText, Music2, Image, Video, Film } from 'lucide-react'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Input } from '@/components/ui/input'
import type { Track, TrackType } from '@/types/timeline'
import { useT } from '@/lib/i18n'

export const TRACK_LABEL_WIDTH = 18   // px
export const TRACK_HEIGHT = 40        // px
export const MAINTAIN_TRACK_HEIGHT = 80 // px (2x normal)

const TRACK_ICONS: Record<TrackType, React.ReactNode> = {
  prompt:   <MessageSquareText className="w-3 h-3 text-background" />,
  audio:    <Music2            className="w-3 h-3 text-background" />,
  image:    <Image             className="w-3 h-3 text-background" />,
  video:    <Video             className="w-3 h-3 text-background" />,
  maintain: <Film              className="w-3 h-3 text-background" />,
}

// Exactly 3 tool slots, rendered as equal-height cells in the sidebar
type ToolSlots = [React.ReactNode?, React.ReactNode?, React.ReactNode?]

interface TrackRowProps {
  track: Track
  children: React.ReactNode
  onTrackChange: (patch: Partial<Track>) => void
  toolSlots?: ToolSlots
  height?: number
  /** When set, use flex-grow instead of fixed height (for proportional sizing) */
  grow?: number
}

export function TrackRow({ track, children, onTrackChange, toolSlots, height, grow }: Readonly<TrackRowProps>) {
  const t = useT()
  const [colorInput, setColorInput] = useState(track.color)
  const rowHeight = height ?? TRACK_HEIGHT

  function commitColor() {
    if (/^#[0-9a-fA-F]{6}$/.test(colorInput)) {
      onTrackChange({ color: colorInput })
    } else {
      setColorInput(track.color)
    }
  }

  return (
    <div
      className="flex border-b border-border"
      style={grow !== undefined
        ? { flexGrow: grow, minHeight: 0 }
        : { height: rowHeight, minHeight: rowHeight }}
    >
      {/* Left sidebar */}
      {/* <div
        className="flex shrink-0 border-r border-border"
        style={{ width: TRACK_LABEL_WIDTH }}
      > */}
        {/* Color swatch column */}
        {/* <div className="flex items-center justify-center border-r border-border" style={{ width: 24 }}> */}
          {/* <Popover>
            <PopoverTrigger asChild>
              <button
                className="w-3 h-3 rounded-sm shrink-0 border border-border focus:outline-none focus:ring-1 focus:ring-ring"
                style={{ backgroundColor: track.color }}
                aria-label="Track color"
              />
            </PopoverTrigger>
            <PopoverContent className="w-44 p-2 space-y-2" align="start" side="right">
              <div className="flex items-center gap-1.5">
                {TRACK_ICONS[track.type]}
                <p className="text-xs font-medium text-foreground">{track.name}</p>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded border border-border"
                  style={{ backgroundColor: colorInput }}
                />
                <Input
                  className="h-6 flex-1 text-xs font-mono px-1"
                  value={colorInput}
                  onChange={(e) => setColorInput(e.target.value)}
                  onBlur={commitColor}
                  onKeyDown={(e) => e.key === 'Enter' && commitColor()}
                  maxLength={7}
                />
              </div>
              <input
                type="color"
                className="w-full h-6 cursor-pointer rounded"
                value={colorInput}
                onChange={(e) => {
                  setColorInput(e.target.value)
                  onTrackChange({ color: e.target.value })
                }}
              />
            </PopoverContent>
          </Popover> */}
          {/* <Tooltip>
            <TooltipTrigger asChild>
               <button
                className="w-3 h-3 rounded-sm shrink-0 border border-border focus:outline-none focus:ring-1 focus:ring-ring"
                style={{ backgroundColor: track.color }}
                aria-label="Track color"
              />
            </TooltipTrigger>
            <TooltipContent>
              <div className="flex items-center gap-1.5 text-background">
                {TRACK_ICONS[track.type]}
                <p className="text-xs font-medium">{track.name}</p>
              </div>
            </TooltipContent>
          </Tooltip> */}
        {/* </div> */}

        {/* 3 tool slots — equal height cells */}
        {/* <div className="flex flex-col flex-1">
          {([0, 1, 2] as const).map((i) => (
            <div
              key={i}
              className="flex-1 flex items-center justify-center border-b border-border/50 last:border-b-0"
            >
              {toolSlots?.[i] ?? null}
            </div>
          ))}
        </div>
      </div> */}

      {/* Track content area */}
      <div className="relative flex-1 overflow-hidden">
        {children}
      </div>
    </div>
  )
}
