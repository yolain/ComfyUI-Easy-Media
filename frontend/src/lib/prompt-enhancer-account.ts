export const PROMPT_ENHANCER_MODELS = {
  minimax: 'h3-context-ir (海螺官方)',
  volcengine: 'doubao-seed-2-0-pro-260215 (火山引擎)',
  zhipu: 'glm-5v-turbo (智谱)',
  runningHubDoubao: 'bytedance/doubao-seed-2.0-pro (RunningHub)',
  runningHubGlm: 'glm-5v-turbo (RunningHub)',
  local: 'llama.cpp (本地)',
} as const

export interface PromptEnhancerBalanceRequest {
  url: string
  refreshIntervalMs: number
  createInit: (apiKey: string) => RequestInit
  readBalance: (payload: unknown) => PromptEnhancerBalance
}

export interface PromptEnhancerBalance {
  amount: number
  currency: string
}

export interface PromptEnhancerAccountProvider {
  id: 'minimax' | 'volcengine' | 'zhipu' | 'runninghub' | 'local'
  modelNames: readonly string[]
  balancePageUrl?: string
  apiKeyPageUrl?: string
  balanceRequest?: PromptEnhancerBalanceRequest
}

interface BalanceCacheEntry {
  balance?: PromptEnhancerBalance
  completedAt?: number
  error?: unknown
  request?: Promise<PromptEnhancerBalance>
}

const balanceCache = new Map<string, BalanceCacheEntry>()

function readRunningHubBalance(payload: unknown): PromptEnhancerBalance {
  if (!payload || typeof payload !== 'object') {
    throw new Error('RunningHub returned an invalid account response.')
  }
  const response = payload as { amount?: unknown; currency?: unknown; error?: unknown }
  if (typeof response.error === 'string') throw new Error(response.error)
  const amount = Number(response.amount)
  if (!Number.isFinite(amount)) {
    throw new Error('RunningHub account response did not contain a valid balance.')
  }
  return {
    amount,
    currency: typeof response.currency === 'string' ? response.currency : 'CNY',
  }
}

/**
 * Single source of truth for prompt-enhancer account endpoints and console links.
 * Add or update a provider here; the account bar derives all of its behavior from it.
 */
export const PROMPT_ENHANCER_ACCOUNT_PROVIDERS: readonly PromptEnhancerAccountProvider[] = [
  {
    id: 'minimax',
    modelNames: [PROMPT_ENHANCER_MODELS.minimax],
    balancePageUrl: 'https://platform.minimaxi.com/console/recharge-records',
    apiKeyPageUrl: 'https://platform.minimaxi.com/console/access?tab=api-keys',
  },
  {
    id: 'volcengine',
    modelNames: [PROMPT_ENHANCER_MODELS.volcengine],
    balancePageUrl: 'https://console.volcengine.com/finance/',
    apiKeyPageUrl: 'https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey',
  },
  {
    id: 'zhipu',
    modelNames: [PROMPT_ENHANCER_MODELS.zhipu],
    balancePageUrl: 'https://open.bigmodel.cn/usercenter/financial/balance',
    apiKeyPageUrl: 'https://open.bigmodel.cn/usercenter/apikeys',
  },
  {
    id: 'runninghub',
    modelNames: [
      PROMPT_ENHANCER_MODELS.runningHubDoubao,
      PROMPT_ENHANCER_MODELS.runningHubGlm,
    ],
    balancePageUrl: 'https://www.runninghub.cn/call-api/bill-task?tab=llmLogs&inviteCode=rh-v1623',
    apiKeyPageUrl: 'https://www.runninghub.cn/zh-cn/enterprise-api/sharedApi?inviteCode=rh-v1623',
    balanceRequest: {
      url: '/easy-media/prompt-enhancer/runninghub-balance',
      refreshIntervalMs: 30_000,
      createInit: (apiKey) => ({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey }),
      }),
      readBalance: readRunningHubBalance,
    },
  },
  {
    id: 'local',
    modelNames: [PROMPT_ENHANCER_MODELS.local],
  },
]

export function getPromptEnhancerAccountProvider(
  modelName: string,
): PromptEnhancerAccountProvider {
  return PROMPT_ENHANCER_ACCOUNT_PROVIDERS.find((provider) => (
    provider.modelNames.includes(modelName)
  )) ?? PROMPT_ENHANCER_ACCOUNT_PROVIDERS[0]
}

/** Shares in-flight requests and serves the last result during the provider cooldown. */
export async function loadPromptEnhancerBalance(
  provider: PromptEnhancerAccountProvider,
  apiKey: string,
): Promise<PromptEnhancerBalance> {
  const balanceRequest = provider.balanceRequest
  if (!balanceRequest) throw new Error('The selected provider does not expose a balance API.')

  const cacheKey = `${provider.id}:${apiKey}`
  const cached = balanceCache.get(cacheKey)
  if (cached?.request) return cached.request
  if (cached?.completedAt !== undefined && Date.now() - cached.completedAt < balanceRequest.refreshIntervalMs) {
    if (cached.balance) return cached.balance
    if (cached.error !== undefined) throw cached.error
  }

  const request = (async () => {
    const response = await fetch(balanceRequest.url, balanceRequest.createInit(apiKey))
    if (!response.ok) throw new Error(`Balance request failed with status ${response.status}.`)
    const payload: unknown = await response.json()
    return balanceRequest.readBalance(payload)
  })()
  balanceCache.set(cacheKey, { request })
  try {
    const balance = await request
    balanceCache.set(cacheKey, { balance, completedAt: Date.now() })
    return balance
  } catch (error) {
    balanceCache.set(cacheKey, { error, completedAt: Date.now() })
    throw error
  }
}

export function clearPromptEnhancerBalanceCache(): void {
  balanceCache.clear()
}
