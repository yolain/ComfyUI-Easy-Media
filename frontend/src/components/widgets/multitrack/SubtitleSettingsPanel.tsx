import { useState } from 'react'
import { Ban, FileAudio, Loader2, Mic2, Plus, Trash2, Type } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ColorPickerPopover } from '@/components/ui/color-picker'
import { NumberInput } from '@/components/ui/number-input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { useT } from '@/lib/i18n'
import { cn } from '@/lib/utils'
import {
  DEFAULT_SUBTITLE_SPEECH_SETTINGS,
  type SubtitleSpeechSettings,
} from '@/lib/subtitle-speech'
import type { MultiTrackSubtitleStyle } from '@/types/multitrack'

interface SubtitleSettingsPanelProps {
  text: string
  style: MultiTrackSubtitleStyle
  onTextChange: (text: string) => void
  onStyleChange: (patch: Partial<MultiTrackSubtitleStyle>) => void
  speechSettings?: SubtitleSpeechSettings
  onSpeechSettingsChange?: (settings: SubtitleSpeechSettings) => void
  onGenerateSpeech?: (settings: SubtitleSpeechSettings) => Promise<void>
}

interface SubtitleStylePreset {
  id: string
  label: string
  style: Pick<MultiTrackSubtitleStyle, 'color' | 'outline_color' | 'background_color'>
}

const SUBTITLE_STYLE_PRESETS: SubtitleStylePreset[] = [
  {
    id: 'no-outline',
    label: 'No outline',
    style: { color: '#ffffff', outline_color: 'transparent', background_color: 'transparent' },
  },
  {
    id: 'white-outline',
    label: 'White',
    style: { color: '#ffffff', outline_color: '#000000', background_color: 'transparent' },
  },
  {
    id: 'yellow-outline',
    label: 'Yellow',
    style: { color: '#ffd60a', outline_color: '#000000', background_color: 'transparent' },
  },
  {
    id: 'black-card',
    label: 'Black',
    style: { color: '#ffffff', outline_color: 'transparent', background_color: '#000000' },
  },
  {
    id: 'white-card',
    label: 'White card',
    style: { color: '#000000', outline_color: 'transparent', background_color: '#ffffff' },
  },
  {
    id: 'accent-card',
    label: 'Accent',
    style: { color: '#ffffff', outline_color: 'transparent', background_color: '#b65764' },
  },
  {
    id: 'green-outline',
    label: 'Green',
    style: { color: '#50f0a0', outline_color: '#000000', background_color: 'transparent' },
  },
  {
    id: 'blue-shadow',
    label: 'Blue',
    style: { color: '#ffffff', outline_color: '#2f8cff', background_color: 'transparent' },
  },
  {
    id: 'pink-shadow',
    label: 'Pink',
    style: { color: '#ffffff', outline_color: '#ff5ba7', background_color: 'transparent' },
  },
  {
    id: 'cream-card',
    label: 'Cream',
    style: { color: '#251f1a', outline_color: 'transparent', background_color: '#f4e8b7' },
  },
  {
    id: 'red-outline',
    label: 'Red',
    style: { color: '#ff334e', outline_color: '#ffffff', background_color: 'transparent' },
  },
  {
    id: 'teal-card',
    label: 'Teal',
    style: { color: '#ffffff', outline_color: 'transparent', background_color: '#0f766e' },
  },
  {
    id: 'rose-card',
    label: 'Rose',
    style: { color: '#fff1f4', outline_color: 'transparent', background_color: '#be4f60' },
  },
]

function normalizePickerColor(value: string | undefined, fallback: string): string {
  if (!value) return fallback
  if (value.trim().toLowerCase() === 'transparent') return 'rgba(0, 0, 0, 0)'
  return value
}

function normalizePresetColor(value: string | undefined): string {
  const normalized = normalizePickerColor(value, 'rgba(0, 0, 0, 0)').replaceAll(' ', '').toLowerCase()
  if (normalized === 'rgba(0,0,0,0)') return 'transparent'
  return normalized
}

function isPresetSelected(style: MultiTrackSubtitleStyle, preset: SubtitleStylePreset): boolean {
  return normalizePresetColor(style.color) === normalizePresetColor(preset.style.color)
    && normalizePresetColor(style.outline_color) === normalizePresetColor(preset.style.outline_color)
    && normalizePresetColor(style.background_color) === normalizePresetColor(preset.style.background_color)
}

function percentValue(value: number): number {
  return Math.round(value * 100)
}

function fromPercentValue(value: number): number {
  return Math.max(0, Math.min(1, value / 100))
}

function toSpeechCfgValue(value: number): number {
  return Number(value.toFixed(1))
}

function hasBackgroundColor(backgroundColor: string): boolean {
  return normalizePresetColor(backgroundColor) !== 'transparent'
}

export function SubtitleSettingsPanel({
  text,
  style,
  onTextChange,
  onStyleChange,
  speechSettings: controlledSpeechSettings,
  onSpeechSettingsChange,
  onGenerateSpeech,
}: Readonly<SubtitleSettingsPanelProps>) {
  const t = useT()
  const [uncontrolledSpeechSettings, setUncontrolledSpeechSettings] = useState<SubtitleSpeechSettings>(DEFAULT_SUBTITLE_SPEECH_SETTINGS)
  const speechSettings = controlledSpeechSettings ?? uncontrolledSpeechSettings
  const [referenceAudioSelectorOpen, setReferenceAudioSelectorOpen] = useState(false)
  const [isGeneratingSpeech, setIsGeneratingSpeech] = useState(false)
  const backgroundOpacity = style.background_opacity ?? 0.7
  const backgroundOpacityPercent = percentValue(backgroundOpacity)
  const backgroundOpacityDisabled = !hasBackgroundColor(style.background_color)

  function updateSpeechSettings(patch: Partial<SubtitleSpeechSettings>) {
    const next = { ...speechSettings, ...patch }
    if (controlledSpeechSettings) {
      onSpeechSettingsChange?.(next)
      return
    }
    setUncontrolledSpeechSettings(next)
  }

  async function handleGenerateSpeech() {
    if (!onGenerateSpeech || isGeneratingSpeech) return
    setIsGeneratingSpeech(true)
    try {
      await onGenerateSpeech(speechSettings)
    } finally {
      setIsGeneratingSpeech(false)
    }
  }

  return (
    <aside
      data-testid="subtitle-settings-panel"
      className="flex h-full min-h-0 w-[35%] min-w-72 shrink-0 flex-col overflow-hidden border-l border-border bg-muted/30 text-card-foreground shadow-lg"
      aria-label={t('multitrack.subtitleTextSettings')}
    >
      <Tabs defaultValue="text" className="flex min-h-0 flex-1 flex-col">
        <div className="border-b border-border p-1">
          <TabsList className="h-8 bg-card border-0 cursor-pointer">
            <TabsTrigger value="text" className="gap-1 text-[10px]">
              <Type className="h-3 w-3" />
              {t('multitrack.text')}
            </TabsTrigger>
            <TabsTrigger value="speech" className="gap-1 text-[10px]">
              <Mic2 className="h-3 w-3" />
              {t('multitrack.speech')}
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="text" className="mt-0 min-h-0 flex-1 overflow-y-auto p-3">
          <div className="grid gap-4">
            <label className="grid gap-1.5 text-xs font-medium text-foreground">
              <Textarea
                aria-label={t('multitrack.subtitleText')}
                className="min-h-20 resize-none bg-card text-xs"
                value={text}
                onChange={(event) => onTextChange(event.currentTarget.value)}
              />
            </label>

            <div className="grid grid-cols-[3rem_minmax(0,1fr)] items-center gap-3">
              <span className="text-[11px] font-medium text-foreground">{t('multitrack.fontSize')}</span>
              <div className="flex min-w-0 items-center gap-2">
                <div className="flex-1 w-full">
                  <Slider
                    aria-label={t('multitrack.fontSize')}
                    value={[style.font_size]}
                    min={8}
                    max={66}
                    step={1}
                    onValueChange={(value) => onStyleChange({ font_size: value[0] ?? style.font_size })}
                  />
                </div>
                <NumberInput
                  aria-label={t('multitrack.fontSize')}
                  className="h-7 w-16 shrink-0"
                  min={8}
                  max={96}
                  step={1}
                  value={style.font_size}
                  onChange={(value) => onStyleChange({ font_size: value })}
                />
              </div>
            </div>

            <div className="grid grid-cols-[3rem_minmax(0,1fr)] items-center gap-3">
              <span className="text-[11px] font-medium text-foreground">{t('multitrack.backgroundOpacity')}</span>
              <div className="flex min-w-0 items-center gap-2">
                <div className="flex-1 w-full">
                  <Slider
                    aria-label={t('multitrack.backgroundOpacity')}
                    value={[backgroundOpacityPercent]}
                    min={0}
                    max={100}
                    step={1}
                    disabled={backgroundOpacityDisabled}
                    onValueChange={(value) => onStyleChange({ background_opacity: fromPercentValue(value[0] ?? backgroundOpacityPercent) })}
                  />
                </div>
                <NumberInput
                  aria-label={t('multitrack.backgroundOpacityPercent')}
                  className="h-7 w-16 shrink-0"
                  min={0}
                  max={100}
                  step={1}
                  disabled={backgroundOpacityDisabled}
                  value={backgroundOpacityPercent}
                  onChange={(value) => onStyleChange({ background_opacity: fromPercentValue(value) })}
                />
              </div>
            </div>

            <div className="grid grid-cols-[3rem_minmax(0,1fr)] items-center gap-3">
              <span className="text-[11px] font-medium text-foreground">{t('multitrack.textColor')}</span>
              <ColorPickerPopover
                value={normalizePickerColor(style.color, '#ffffff')}
                defaultFormat="hex"
                triggerAriaLabel={t('multitrack.textColor')}
                triggerClassName="w-full justify-start h-6 text-[10px]"
                side="left"
                align="start"
                sideOffset={10}
                onValueChange={(value) => onStyleChange({ color: value })}
              />
            </div>

            {/* <div className="grid gap-2 text-xs font-medium text-foreground">
              <span>{t('multitrack.outlineColor')}</span>
              <ColorPickerPopover
                value={normalizePickerColor(style.outline_color, '#000000')}
                defaultFormat="hex"
                triggerAriaLabel={t('multitrack.outlineColor')}
                triggerClassName="w-full justify-start"
                side="left"
                align="start"
                sideOffset={10}
                onValueChange={(value) => onStyleChange({ outline_color: value })}
              />
            </div> */}

            {/* <div className="grid gap-2 text-xs font-medium text-foreground">
              <span>{t('multitrack.backgroundColor')}</span>
              <ColorPickerPopover
                value={normalizePickerColor(style.background_color, 'rgba(0, 0, 0, 0)')}
                defaultFormat="rgb"
                triggerAriaLabel={t('multitrack.backgroundColor')}
                triggerClassName="w-full justify-start"
                side="left"
                align="start"
                sideOffset={10}
                onValueChange={(value) => onStyleChange({ background_color: value })}
              />
            </div> */}

            <div className="grid gap-2">
              <span className="text-[11px] font-medium text-foreground">{t('multitrack.presetStyle')}</span>
              <div className="grid grid-cols-7 gap-1.5">
                {SUBTITLE_STYLE_PRESETS.map((preset) => {
                  const selected = isPresetSelected(style, preset)
                  return (
                    <Button
                      key={preset.id}
                      type="button"
                      variant="ghost"
                      aria-pressed={selected}
                      className={cn(
                        'flex aspect-square h-auto min-h-0 w-full items-center justify-center rounded-md border bg-muted p-0 text-base font-bold transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                        selected ? 'border-primary ring-2 ring-primary' : 'border-border',
                      )}
                      aria-label={t('multitrack.applyPresetStyle', { name: preset.label })}
                      onClick={() => onStyleChange(preset.style)}
                    >
                      {preset.id === 'no-outline' ? (
                        <Ban className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <span
                          className="rounded px-1 leading-none"
                          style={{
                            color: preset.style.color,
                            backgroundColor: preset.style.background_color,
                            textShadow: `1px 0 0 ${preset.style.outline_color}, -1px 0 0 ${preset.style.outline_color}, 0 1px 0 ${preset.style.outline_color}, 0 -1px 0 ${preset.style.outline_color}`,
                          }}
                        >
                          T
                        </span>
                      )}
                    </Button>
                  )
                })}
              </div>
            </div>

            <div className="grid grid-cols-[3rem_repeat(3,minmax(0,1fr))] items-center gap-2">
              <span className="text-[11px] font-medium text-foreground">{t('multitrack.position')}%</span>
              <label className="flex min-w-0 items-center gap-1">
                <span className="text-[10px] font-medium text-muted-foreground">{t('multitrack.positionX')}</span>
                <NumberInput
                  aria-label={t('multitrack.positionXPercent')}
                  className="h-7 min-w-0 flex-1"
                  min={0}
                  max={100}
                  step={1}
                  value={percentValue(style.x)}
                  onChange={(value) => onStyleChange({ x: fromPercentValue(value) })}
                />
              </label>
              <label className="flex min-w-0 items-center gap-1">
                <span className="text-[10px] font-medium text-muted-foreground">{t('multitrack.positionY')}</span>
                <NumberInput
                  aria-label={t('multitrack.positionYPercent')}
                  className="h-7 min-w-0 flex-1"
                  min={0}
                  max={95}
                  step={1}
                  value={percentValue(style.y)}
                  onChange={(value) => onStyleChange({ y: Math.min(0.95, fromPercentValue(value)) })}
                />
              </label>
              <label className="flex min-w-0 items-center gap-1">
                <span className="text-[10px] font-medium text-muted-foreground">{t('multitrack.subtitleWidthShort')}</span>
                <NumberInput
                  aria-label={t('multitrack.subtitleWidthPercent')}
                  className="h-7 min-w-0 flex-1"
                  min={10}
                  max={100}
                  step={1}
                  value={percentValue(style.width)}
                  onChange={(value) => onStyleChange({ width: Math.max(0.1, fromPercentValue(value)) })}
                />
              </label>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="speech" className="mt-0 flex min-h-0 flex-1 flex-col overflow-hidden p-0">
          <div className="min-h-0 flex-1 overflow-y-auto p-3 pb-4">
            <div className="grid gap-4">
              <div className="grid grid-cols-[4.25rem_minmax(0,1fr)] items-center gap-3">
                <span className="text-[11px] font-medium text-foreground">{t('multitrack.speechModel')}</span>
                <Select
                  value={speechSettings.model}
                  onValueChange={(value) => updateSpeechSettings({ model: value as SubtitleSpeechSettings['model'] })}
                >
                  <SelectTrigger
                    aria-label={t('multitrack.speechModel')}
                    className="h-8 w-full bg-card text-xs"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="VoxCPM2">VoxCPM2</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <label className="grid gap-1.5">
                <Textarea
                  aria-label={t('multitrack.speechPrompt')}
                  className="min-h-20 resize-none bg-card text-xs"
                  value={speechSettings.prompt}
                  placeholder={t('multitrack.speechPromptPlaceholder')}
                  onChange={(event) => updateSpeechSettings({ prompt: event.currentTarget.value })}
                />
              </label>

              <div className="grid grid-cols-[4.25rem_minmax(0,1fr)] items-center gap-3">
                <span className="text-[11px] font-medium text-foreground">{t('multitrack.speechCfg')}</span>
                <div className="flex min-w-0 items-center gap-2">
                  <div className="flex-1 w-full">
                    <Slider
                      aria-label={t('multitrack.speechCfg')}
                      value={[speechSettings.cfg]}
                      min={0.1}
                      max={5}
                      step={0.1}
                      onValueChange={(value) => updateSpeechSettings({ cfg: toSpeechCfgValue(value[0] ?? speechSettings.cfg) })}
                    />
                  </div>
                  <NumberInput
                    aria-label={t('multitrack.speechCfg')}
                    className="h-7 w-16 shrink-0"
                    min={0.1}
                    max={5}
                    step={0.1}
                    value={speechSettings.cfg}
                    formatValue={(value) => value.toFixed(1)}
                    onChange={(value) => updateSpeechSettings({ cfg: toSpeechCfgValue(value) })}
                  />
                </div>
              </div>

              <div className="grid grid-cols-[4.25rem_minmax(0,1fr)] items-center gap-3">
                <span className="text-[11px] font-medium text-foreground">{t('multitrack.speechSteps')}</span>
                <div className="flex min-w-0 items-center gap-2">
                  <div className="flex-1 w-full">
                    <Slider
                      aria-label={t('multitrack.speechSteps')}
                      value={[speechSettings.steps]}
                      min={1}
                      max={50}
                      step={1}
                      onValueChange={(value) => updateSpeechSettings({ steps: value[0] ?? speechSettings.steps })}
                    />
                  </div>
                  <NumberInput
                    aria-label={t('multitrack.speechSteps')}
                    className="h-7 w-16 shrink-0"
                    min={1}
                    max={50}
                    step={1}
                    value={speechSettings.steps}
                    onChange={(value) => updateSpeechSettings({ steps: value })}
                  />
                </div>
              </div>

              <div className="grid grid-cols-[4.25rem_minmax(0,1fr)] items-start gap-3">
                <span className="pt-1.5 text-[11px] font-medium text-foreground">{t('multitrack.referenceAudio')}</span>
                <div className="min-w-0">
                  {speechSettings.referenceAudio ? (
                    <div className="flex min-w-0 items-center gap-2">
                      <Popover open={referenceAudioSelectorOpen} onOpenChange={setReferenceAudioSelectorOpen}>
                        <PopoverTrigger asChild>
                          <Button
                            type="button"
                            variant="secondary"
                            className="h-8 min-w-0 flex-1 cursor-pointer justify-start bg-card px-2 text-xs"
                            aria-label={t('multitrack.reselectReferenceAudio')}
                          >
                            <FileAudio className="h-3.5 w-3.5 shrink-0 text-highlight" />
                            <span className="truncate">{speechSettings.referenceAudio}</span>
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="end">
                          <MediaSelector
                            value={speechSettings.referenceAudio}
                            mediaType="audio"
                            defaultTab="inputs"
                            onChange={(filePath, source) => {
                              updateSpeechSettings({
                                referenceAudio: filePath,
                                referenceAudioSourceType: source ?? 'input',
                              })
                              setReferenceAudioSelectorOpen(false)
                            }}
                          />
                        </PopoverContent>
                      </Popover>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0 cursor-pointer text-destructive"
                        aria-label={t('multitrack.clearReferenceAudio')}
                        onClick={() => updateSpeechSettings({ referenceAudio: '' })}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  ) : (
                    <Popover open={referenceAudioSelectorOpen} onOpenChange={setReferenceAudioSelectorOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          type="button"
                          variant="secondary"
                          className="h-8 w-full cursor-pointer justify-start bg-card text-xs"
                        >
                          <Plus className="h-3.5 w-3.5" />
                          {t('multitrack.addReferenceAudio')}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0" align="end">
                        <MediaSelector
                          value=""
                          mediaType="audio"
                          defaultTab="inputs"
                          onChange={(filePath, source) => {
                            updateSpeechSettings({
                              referenceAudio: filePath,
                              referenceAudioSourceType: source ?? 'input',
                            })
                            setReferenceAudioSelectorOpen(false)
                          }}
                        />
                      </PopoverContent>
                    </Popover>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="sticky bottom-0 flex shrink-0 justify-end border-t border-border bg-muted/80 p-2 backdrop-blur">
            <Button
              type="button"
              disabled={!onGenerateSpeech || isGeneratingSpeech}
              className="h-7 cursor-pointer bg-highlight px-3 text-[11px] text-background shadow hover:bg-highlight/90"
              onClick={handleGenerateSpeech}
            >
              {isGeneratingSpeech ? <Loader2 className="h-3 w-3 animate-spin" /> : <Mic2 className="h-3 w-3" />}
              {t('multitrack.speech')}
            </Button>
          </div>
        </TabsContent>
      </Tabs>
    </aside>
  )
}
