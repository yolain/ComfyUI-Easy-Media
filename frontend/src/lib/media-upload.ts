export async function uploadInputMediaFile(file: File): Promise<string> {
  const form = new FormData()
  form.append('image', file)
  form.append('type', 'input')
  form.append('overwrite', 'false')
  const response = await fetch('/upload/image', { method: 'POST', body: form })
  if (!response.ok) throw new Error(`Upload failed: ${response.status}`)
  const result = await response.json() as { name: string; subfolder?: string }
  return result.subfolder ? `${result.subfolder}/${result.name}` : result.name
}
