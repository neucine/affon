export type Routing = Readonly<{
  schema_version: 1
  backend: "zig" | "cyna"
  configured_backend: "cpu" | "metal" | "cuda"
  category: "success" | "configuration" | "backend" | "observability" | "lowering" | "operation" | "dense_graph"
  reason: string
  operation: string | null
  option: string | null
}>

export type RoutingEntry = Readonly<{
  name: string
  routing: Routing
}>

export type FallbackClassification = "portable_lowering" | "observability" | "execution_limit" | "unclassified"

export type FallbackGroup = Readonly<{
  classification: FallbackClassification
  category: Routing["category"]
  reason: string
  operation: string | null
  option: string | null
  program_count: number
  programs: readonly string[]
}>

export type RoutingReport = Readonly<{
  schema_version: 1
  suite: "affon_cyna_routing"
  programs: Readonly<{
    total: number
    cyna: number
    zig: number
    cyna_percent: number
  }>
  fallback_groups: readonly FallbackGroup[]
  highest_impact_portable_lowering: FallbackGroup | null
}>

function classification(category: Routing["category"]): FallbackClassification {
  switch (category) {
    case "operation":
    case "dense_graph":
      return "portable_lowering"
    case "lowering":
      return "unclassified"
    case "observability":
      return "observability"
    case "backend":
    case "configuration":
      return "execution_limit"
    default:
      throw new TypeError(`unexpected Zig fallback category: ${category}`)
  }
}

function routeKey(category: Routing["category"], reason: string, operation: string | null, option: string | null): string {
  return JSON.stringify([category, reason, operation, option])
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function compareOptionalText(left: string | null, right: string | null): number {
  return compareText(left ?? "", right ?? "")
}

function compareGroups(left: FallbackGroup, right: FallbackGroup): number {
  return (right.program_count - left.program_count)
    || compareText(left.classification, right.classification)
    || compareText(left.category, right.category)
    || compareText(left.reason, right.reason)
    || compareOptionalText(left.operation, right.operation)
    || compareOptionalText(left.option, right.option)
}

export function buildRoutingReport(entries: readonly RoutingEntry[]): RoutingReport {
  let cyna = 0
  let zig = 0
  const groups = new Map<string, { classification: FallbackClassification; category: Routing["category"]; reason: string; operation: string | null; option: string | null; programs: string[] }>()
  for (const entry of entries) {
    if (entry.routing.backend === "cyna") {
      cyna += 1
      continue
    }
    zig += 1
    const fallbackClassification = classification(entry.routing.category)
    const key = routeKey(entry.routing.category, entry.routing.reason, entry.routing.operation, entry.routing.option)
    let group = groups.get(key)
    if (!group) {
      group = {
        classification: fallbackClassification,
        category: entry.routing.category,
        reason: entry.routing.reason,
        operation: entry.routing.operation,
        option: entry.routing.option,
        programs: [],
      }
      groups.set(key, group)
    }
    group.programs.push(entry.name)
  }

  const fallbackGroups = [...groups.values()].map(group => Object.freeze({
    ...group,
    program_count: group.programs.length,
    programs: Object.freeze(group.programs.slice().sort(compareText)),
  })).sort(compareGroups)
  const total = entries.length
  return Object.freeze({
    schema_version: 1,
    suite: "affon_cyna_routing",
    programs: Object.freeze({
      total,
      cyna,
      zig,
      cyna_percent: total === 0 ? 0 : (cyna * 100) / total,
    }),
    fallback_groups: Object.freeze(fallbackGroups.map(group => Object.freeze(group))),
    highest_impact_portable_lowering: fallbackGroups.find(group => group.classification === "portable_lowering") ?? null,
  })
}
