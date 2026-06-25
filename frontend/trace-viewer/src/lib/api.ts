export interface TraceSummaryDTO {
  id: string;
  name: string;
  kind: "prod";
  createdAt: number;
  status: "success" | "error";
  spansCount: number;
  durationMs: number;
}

export interface EvalRunDTO {
  run: string;
  reportId: string;
  packId: string;
  createdAt: number;
  scorer: string;
  executor: string;
  samples: number;
  caseCount: number;
  passedCount: number;
  failedCount: number;
}

export interface RunListResponseDTO {
  runs: EvalRunDTO[];
}

export interface EvalTraceSummaryDTO {
  id: string;
  name: string;
  kind: "eval";
  status: "success" | "error" | "warning";
  score: number | null;
  passCount: number;
  sampleCount: number;
  goldEpisodeId: string;
  spansCount: number;
}

export interface TraceListResponseDTO {
  traces: TraceSummaryDTO[];
  total: number;
}

export interface EvalTraceListResponseDTO {
  traces: EvalTraceSummaryDTO[];
  total: number;
}

export interface SpanAttributeDTO {
  key: string;
  value: {
    stringValue?: string;
  };
}

export interface SpanDTO {
  id: string;
  title: string;
  startTimeMs: number;
  endTimeMs: number;
  durationMs: number;
  type: string;
  status: string;
  input: unknown;
  output: unknown;
  raw: unknown;
  attributes: SpanAttributeDTO[];
  children: SpanDTO[];
}

export interface TraceRecordDTO {
  id: string;
  name: string;
  spansCount: number;
  durationMs: number;
  agentDescription: string;
  startTimeMs: number;
}

export interface TraceBadgeDTO {
  label: string;
}

export interface TraceDetailDTO {
  traceRecord: TraceRecordDTO;
  spans: SpanDTO[];
  goldEpisodeId?: string;
  badges?: TraceBadgeDTO[];
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export function listTraces(q = ""): Promise<TraceListResponseDTO> {
  const params = new URLSearchParams({
    source: "prod",
    q,
    limit: "50",
    offset: "0",
  });

  return getJson<TraceListResponseDTO>(`/api/traces?${params.toString()}`);
}

export function getTrace(id: string): Promise<TraceDetailDTO> {
  return getJson<TraceDetailDTO>(`/api/traces/${encodeURIComponent(id)}`);
}

export function listRuns(): Promise<RunListResponseDTO> {
  return getJson<RunListResponseDTO>("/api/runs");
}

export function listEvalTraces(run: string): Promise<EvalTraceListResponseDTO> {
  const params = new URLSearchParams({ run });

  return getJson<EvalTraceListResponseDTO>(
    `/api/eval-traces?${params.toString()}`,
  );
}

export function getEvalTrace(
  caseId: string,
  run: string,
): Promise<TraceDetailDTO> {
  const params = new URLSearchParams({ run });

  return getJson<TraceDetailDTO>(
    `/api/eval-traces/${encodeURIComponent(caseId)}?${params.toString()}`,
  );
}
