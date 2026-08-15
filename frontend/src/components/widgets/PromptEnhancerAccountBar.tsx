import { useEffect, useState } from 'react'
import { ExternalLink, KeyRound, LoaderCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { LocaleContext, useT } from '@/lib/i18n'
import {
  getPromptEnhancerAccountProvider,
  loadPromptEnhancerBalance,
  type PromptEnhancerBalance,
} from '@/lib/prompt-enhancer-account'
import {
  readPromptEnhancerAccountSelection,
  type PromptEnhancerAccountNode,
} from '@/lib/prompt-enhancer-account-node'
import type { ReactWidgetProps } from '@/lib/create-react-widget'

interface PromptEnhancerAccountBarProps {
  apiKey: string
  locale?: string
  modelName: string
  refreshToken?: number
}

function openExternal(url: string): void {
  globalThis.open(url, '_blank', 'noopener,noreferrer')
}

function formatBalance(balance: PromptEnhancerBalance, locale?: string): string {
  try {
    return new Intl.NumberFormat(locale || undefined, {
      style: 'currency',
      currency: balance.currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(balance.amount)
  } catch (error) {
    console.error('[PromptEnhancerAccountBar] failed to format balance:', error)
    return `${balance.amount.toFixed(2)} ${balance.currency}`
  }
}

function PromptEnhancerAccountBarContent({
  apiKey,
  locale,
  modelName,
  refreshToken = 0,
}: Readonly<PromptEnhancerAccountBarProps>) {
  const t = useT()
  const provider = getPromptEnhancerAccountProvider(modelName)
  const [balance, setBalance] = useState<PromptEnhancerBalance | null>(null)
  const [balanceError, setBalanceError] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const balanceRequest = provider.balanceRequest
    const trimmedApiKey = apiKey.trim()
    setBalance(null)
    setBalanceError(false)
    setLoading(Boolean(balanceRequest))
    if (!balanceRequest) return

    let cancelled = false
    const loadBalance = async () => {
      try {
        const nextBalance = await loadPromptEnhancerBalance(provider, trimmedApiKey)
        if (!cancelled) setBalance(nextBalance)
      } catch (error) {
        if (cancelled) return
        console.error('[PromptEnhancerAccountBar] failed to load balance:', error)
        setBalanceError(true)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void loadBalance()
    return () => {
      cancelled = true
    }
  }, [apiKey, provider, refreshToken])

  if (provider.id === 'local') {
    return (
      <div className="flex h-[30px] items-center border-t border-border px-2 text-[10px] text-muted-foreground">
        {t('promptEnhancerAccount.localModel')}
      </div>
    )
  }

  const renderBalance = () => {
    if (!provider.balanceRequest) {
      return (
        <Button
          className="h-6 px-1.5 text-[10px] cursor-pointer"
          onClick={() => provider.balancePageUrl && openExternal(provider.balancePageUrl)}
          size="sm"
          type="button"
          variant="link"
        >
          {t('promptEnhancerAccount.viewBalance')}
        </Button>
      )
    }
    if (loading) {
      return (
        <Badge className="h-5 gap-1 px-1.5 text-[10px]" variant="outline">
          <LoaderCircle className="size-3 animate-spin" />
          {t('promptEnhancerAccount.loadingBalance')}
        </Badge>
      )
    }
    if (balance) {
      return (
        <Badge className="h-5 max-w-full truncate px-1.5 text-[10px] cursor-pointer" title={formatBalance(balance, locale)} variant="secondary" onClick={() => provider.balancePageUrl && openExternal(provider.balancePageUrl)}>
          {t('promptEnhancerAccount.balance', { amount: formatBalance(balance, locale) })}
        </Badge>
      )
    }
    const label = balanceError
      ? t('promptEnhancerAccount.balanceUnavailable')
      : t('promptEnhancerAccount.apiKeyRequired')
    return (
      <Badge className="h-5 px-1.5 text-[10px]" title={label} variant="outline">
        {t('promptEnhancerAccount.balanceEmpty')}
      </Badge>
    )
  }

  return (
    <div className="flex h-[30px] min-w-0 items-center justify-between gap-2">
      <div className="min-w-0">{renderBalance()}</div>
      {provider.apiKeyPageUrl ? (
        <Button
          className="h-6 shrink-0 px-1.5 text-[10px] cursor-pointer"
          onClick={() => provider.apiKeyPageUrl && openExternal(provider.apiKeyPageUrl)}
          size="sm"
          type="button"
          variant="outline"
        >
          <KeyRound className="size-1" />
          {t('promptEnhancerAccount.apiKey')}
        </Button>
      ) : null}
    </div>
  )
}

export function PromptEnhancerAccountBar(props: Readonly<PromptEnhancerAccountBarProps>) {
  return (
    <LocaleContext.Provider value={props.locale}>
      <PromptEnhancerAccountBarContent {...props} />
    </LocaleContext.Provider>
  )
}

export function PromptEnhancerAccountWidget({
  app,
  node,
}: Readonly<ReactWidgetProps<string>>) {
  const accountNode = node as PromptEnhancerAccountNode
  const [selection, setSelection] = useState(() => readPromptEnhancerAccountSelection(accountNode))
  const [refreshToken, setRefreshToken] = useState(0)

  useEffect(() => {
    let active = true
    const originalOnConfigure = accountNode.onConfigure
    const originalOnExecuted = accountNode.onExecuted
    const originalOnWidgetChanged = accountNode.onWidgetChanged

    const syncSelection = () => {
      queueMicrotask(() => {
        if (!active) return
        const next = readPromptEnhancerAccountSelection(accountNode)
        setSelection((current) => (
          current.modelName === next.modelName && current.apiKey === next.apiKey
            ? current
            : next
        ))
      })
    }
    const wrappedOnConfigure = function (this: unknown, serialisedNode: unknown) {
      originalOnConfigure?.call(this, serialisedNode)
      syncSelection()
    }
    const wrappedOnExecuted = function (this: unknown, output: unknown) {
      originalOnExecuted?.call(this, output)
      const current = readPromptEnhancerAccountSelection(accountNode)
      if (getPromptEnhancerAccountProvider(current.modelName).id === 'runninghub') {
        setRefreshToken((token) => token + 1)
      }
    }
    const wrappedOnWidgetChanged = function (
      this: unknown,
      name: string,
      value: unknown,
      oldValue: unknown,
      widget: unknown,
    ) {
      originalOnWidgetChanged?.call(this, name, value, oldValue, widget)
      if (name === 'model' || name === 'model.model' || name === 'model.apikey' || name === 'apikey') {
        syncSelection()
      }
    }

    accountNode.onConfigure = wrappedOnConfigure
    accountNode.onExecuted = wrappedOnExecuted
    accountNode.onWidgetChanged = wrappedOnWidgetChanged
    syncSelection()
    return () => {
      active = false
      if (accountNode.onConfigure === wrappedOnConfigure) accountNode.onConfigure = originalOnConfigure
      if (accountNode.onExecuted === wrappedOnExecuted) accountNode.onExecuted = originalOnExecuted
      if (accountNode.onWidgetChanged === wrappedOnWidgetChanged) {
        accountNode.onWidgetChanged = originalOnWidgetChanged
      }
    }
  }, [accountNode])

  return (
    <PromptEnhancerAccountBar
      apiKey={selection.apiKey}
      locale={app?.ui?.settings?.settingsValues?.['Comfy.Locale']}
      modelName={selection.modelName}
      refreshToken={refreshToken}
    />
  )
}
