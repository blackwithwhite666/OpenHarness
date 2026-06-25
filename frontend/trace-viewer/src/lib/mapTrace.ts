import type {
  TraceRecord,
  TraceSpan,
  TraceSpanCategory,
  TraceSpanStatus,
} from "@evilmartians/agent-prism-types";

import type {
  SpanDTO,
  TraceBadgeDTO,
  TraceDetailDTO,
  TraceSummaryDTO,
} from "./api";

export type TraceRecordWithBadges = TraceRecord & {
  badges?: TraceBadgeDTO[];
};

export interface MappedTrace {
  traceRecord: TraceRecordWithBadges;
  spans: TraceSpan[];
  goldEpisodeId?: string;
  badges?: TraceBadgeDTO[];
}

const SPAN_TYPES = new Set<TraceSpanCategory>([
  "llm_call",
  "tool_execution",
  "agent_invocation",
  "chain_operation",
  "retrieval",
  "embedding",
  "create_agent",
  "span",
  "event",
  "guardrail",
  "unknown",
]);

const SPAN_STATUSES = new Set<TraceSpanStatus>([
  "success",
  "error",
  "pending",
  "warning",
]);

function mapSpanType(type: string): TraceSpanCategory {
  return SPAN_TYPES.has(type as TraceSpanCategory)
    ? (type as TraceSpanCategory)
    : "unknown";
}

function mapSpanStatus(status: string): TraceSpanStatus {
  if (status === "running") return "pending";

  return SPAN_STATUSES.has(status as TraceSpanStatus)
    ? (status as TraceSpanStatus)
    : "success";
}

function stringifyValue(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return value;

  try {
    const serialized = JSON.stringify(value, null, 2);
    return serialized === undefined ? String(value) : serialized;
  } catch {
    return String(value);
  }
}

export function mapSpan(dto: SpanDTO): TraceSpan {
  return {
    ...dto,
    type: mapSpanType(dto.type),
    status: mapSpanStatus(dto.status),
    input: stringifyValue(dto.input),
    output: stringifyValue(dto.output),
    raw: stringifyValue(dto.raw) ?? JSON.stringify(dto, null, 2),
    startTime: new Date(dto.startTimeMs),
    endTime: new Date(dto.endTimeMs),
    duration: dto.durationMs,
    children: dto.children.map(mapSpan),
  };
}

export function mapTraceRecord(dto: TraceDetailDTO["traceRecord"]): TraceRecord {
  return {
    ...dto,
    startTime: dto.startTimeMs,
  };
}

export function mapTraceSummary(dto: TraceSummaryDTO): TraceRecord {
  return {
    id: dto.id,
    name: dto.name,
    spansCount: dto.spansCount,
    durationMs: dto.durationMs,
    agentDescription: "",
    startTime: dto.createdAt,
  };
}

export function mapTrace(dto: TraceDetailDTO): MappedTrace {
  const badges = dto.badges ?? [];

  return {
    traceRecord: {
      ...mapTraceRecord(dto.traceRecord),
      badges,
    },
    spans: dto.spans.map(mapSpan),
    badges,
    goldEpisodeId: dto.goldEpisodeId,
  };
}
