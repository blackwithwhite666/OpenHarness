export interface TraceSummaryDTO {
  id: string;
  name: string;
  kind: "prod";
  createdAt: number;
  status: "success" | "error";
  spansCount: number;
  durationMs: number;
}

export interface TraceListResponseDTO {
  traces: TraceSummaryDTO[];
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

export interface TraceDetailDTO {
  traceRecord: TraceRecordDTO;
  spans: SpanDTO[];
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
