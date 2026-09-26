/** Stable softmax over all ImageNet classes, followed by top-five ranking. */
export function rank_classes(logits: number[], labels: Record<string, string>) {
  if (!logits.length || logits.some((value) => !Number.isFinite(value)))
    throw new Error('Invalid classifier output')
  const maximum = Math.max(...logits)
  const weights = logits.map((value) => Math.exp(value - maximum))
  const total = weights.reduce((sum, value) => sum + value, 0)
  return weights
    .map((value, id) => ({
      id,
      label: labels[String(id)] ?? `Class ${id}`,
      score: value / total,
    }))
    .sort((a, b) => b.score - a.score || a.id - b.id)
    .slice(0, 5)
}
