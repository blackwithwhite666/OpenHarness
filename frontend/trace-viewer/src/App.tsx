import type { TraceRecord, TraceSpan } from "@evilmartians/agent-prism-types";

import { flattenSpans, filterSpansRecursively } from "@evilmartians/agent-prism-data";
import { AlertCircle, ExternalLink, RefreshCw, Search } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";

import type { BadgeProps } from "./components/agent-prism/Badge";

import { Button } from "./components/agent-prism/Button";
import { DetailsView } from "./components/agent-prism/DetailsView/DetailsView";
import { TextInput } from "./components/agent-prism/TextInput";
import { TraceList } from "./components/agent-prism/TraceList/TraceList";
import { TraceViewerTreeViewContainer } from "./components/agent-prism/TraceViewer/TraceViewerTreeViewContainer";
import {
  getEvalTrace,
  getTrace,
  listEvalTraces,
  listRuns,
  listTraces,
  type EvalRunDTO,
  type EvalTraceSummaryDTO,
} from "./lib/api";
import { mapTrace, mapTraceSummary } from "./lib/mapTrace";

type SourceTab = "prod" | "eval";

type TraceRecordWithBadges = TraceRecord & {
  badges?: BadgeProps[];
};

interface LoadedTrace {
  traceRecord: TraceRecordWithBadges;
  spans: TraceSpan[];
  goldEpisodeId?: string;
  badges?: BadgeProps[];
}

function App() {
  const [sourceTab, setSourceTab] = useState<SourceTab>("prod");
  const [query, setQuery] = useState("");
  const [traces, setTraces] = useState<TraceRecord[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<TraceRecord | undefined>();
  const [loadedTrace, setLoadedTrace] = useState<LoadedTrace | undefined>();
  const [selectedSpan, setSelectedSpan] = useState<TraceSpan | undefined>();
  const [traceListExpanded, setTraceListExpanded] = useState(true);
  const [expandedSpansIds, setExpandedSpansIds] = useState<string[]>([]);
  const [spanSearchValue, setSpanSearchValue] = useState("");
  const [isListLoading, setIsListLoading] = useState(false);
  const [isTraceLoading, setIsTraceLoading] = useState(false);
  const [listError, setListError] = useState<string | undefined>();
  const [traceError, setTraceError] = useState<string | undefined>();
  const [evalRuns, setEvalRuns] = useState<EvalRunDTO[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>();
  const [evalTraces, setEvalTraces] = useState<TraceRecordWithBadges[]>([]);
  const [selectedEvalTrace, setSelectedEvalTrace] =
    useState<TraceRecordWithBadges | undefined>();
  const [loadedEvalTrace, setLoadedEvalTrace] = useState<LoadedTrace | undefined>();
  const [selectedEvalSpan, setSelectedEvalSpan] = useState<TraceSpan | undefined>();
  const [evalTraceListExpanded, setEvalTraceListExpanded] = useState(true);
  const [evalExpandedSpansIds, setEvalExpandedSpansIds] = useState<string[]>([]);
  const [evalSpanSearchValue, setEvalSpanSearchValue] = useState("");
  const [isRunLoading, setIsRunLoading] = useState(false);
  const [isEvalListLoading, setIsEvalListLoading] = useState(false);
  const [isEvalTraceLoading, setIsEvalTraceLoading] = useState(false);
  const [runError, setRunError] = useState<string | undefined>();
  const [evalListError, setEvalListError] = useState<string | undefined>();
  const [evalTraceError, setEvalTraceError] = useState<string | undefined>();

  const fetchTraceList = useCallback(async (search: string) => {
    setIsListLoading(true);
    setListError(undefined);

    try {
      const response = await listTraces(search);
      const nextTraces = response.traces.map(mapTraceSummary);
      setTraces(nextTraces);
    } catch (error) {
      setListError(error instanceof Error ? error.message : "Failed to load traces");
    } finally {
      setIsListLoading(false);
    }
  }, []);

  const clearEvalTraceSelection = useCallback(() => {
    setSelectedEvalTrace(undefined);
    setLoadedEvalTrace(undefined);
    setSelectedEvalSpan(undefined);
    setEvalSpanSearchValue("");
    setEvalExpandedSpansIds([]);
    setEvalTraceError(undefined);
  }, []);

  const fetchEvalRuns = useCallback(async () => {
    setIsRunLoading(true);
    setRunError(undefined);

    try {
      const response = await listRuns();
      setEvalRuns(response.runs);

      if (response.runs.length === 0) {
        setSelectedRunId(undefined);
        setEvalTraces([]);
        clearEvalTraceSelection();
        return;
      }

      setSelectedRunId((currentRunId) =>
        currentRunId && response.runs.some((run) => run.run === currentRunId)
          ? currentRunId
          : response.runs[0].run,
      );
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Failed to load eval runs");
    } finally {
      setIsRunLoading(false);
    }
  }, [clearEvalTraceSelection]);

  const fetchEvalTraceList = useCallback(
    async (run: string) => {
      setIsEvalListLoading(true);
      setEvalListError(undefined);

      try {
        const response = await listEvalTraces(run);
        setEvalTraces(response.traces.map(mapEvalTraceSummary));
      } catch (error) {
        setEvalListError(
          error instanceof Error ? error.message : "Failed to load eval traces",
        );
      } finally {
        setIsEvalListLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (sourceTab !== "prod") return;

    const handle = window.setTimeout(() => {
      void fetchTraceList(query);
    }, 250);

    return () => window.clearTimeout(handle);
  }, [fetchTraceList, query, sourceTab]);

  useEffect(() => {
    if (sourceTab !== "eval") return;

    void fetchEvalRuns();
  }, [fetchEvalRuns, sourceTab]);

  useEffect(() => {
    setEvalTraces([]);
    clearEvalTraceSelection();
  }, [clearEvalTraceSelection, selectedRunId]);

  useEffect(() => {
    if (sourceTab !== "eval" || !selectedRunId) return;

    void fetchEvalTraceList(selectedRunId);
  }, [fetchEvalTraceList, selectedRunId, sourceTab]);

  const allSpanIds = useMemo(
    () => (loadedTrace ? flattenSpans(loadedTrace.spans).map((span) => span.id) : []),
    [loadedTrace],
  );

  const filteredSpans = useMemo(() => {
    if (!loadedTrace) return [];

    return spanSearchValue.trim()
      ? filterSpansRecursively(loadedTrace.spans, spanSearchValue)
      : loadedTrace.spans;
  }, [loadedTrace, spanSearchValue]);

  const handleTraceSelect = useCallback(async (trace: TraceRecord) => {
    setSelectedTrace(trace);
    setLoadedTrace(undefined);
    setSelectedSpan(undefined);
    setSpanSearchValue("");
    setExpandedSpansIds([]);
    setTraceError(undefined);
    setIsTraceLoading(true);

    try {
      const response = await getTrace(trace.id);
      const mappedTrace = mapTrace(response);
      setLoadedTrace(mappedTrace);
      const flatSpans = flattenSpans(mappedTrace.spans);
      setExpandedSpansIds(flatSpans.map((span) => span.id));
      setSelectedSpan(flatSpans[0]);
    } catch (error) {
      setTraceError(error instanceof Error ? error.message : "Failed to load trace");
    } finally {
      setIsTraceLoading(false);
    }
  }, []);

  const handleEvalRunSelect = useCallback(
    (run: string) => {
      setSelectedRunId(run || undefined);
      setEvalTraces([]);
      clearEvalTraceSelection();
    },
    [clearEvalTraceSelection],
  );

  const handleEvalTraceSelect = useCallback(
    async (trace: TraceRecordWithBadges) => {
      if (!selectedRunId) return;

      setSelectedEvalTrace(trace);
      setLoadedEvalTrace(undefined);
      setSelectedEvalSpan(undefined);
      setEvalSpanSearchValue("");
      setEvalExpandedSpansIds([]);
      setEvalTraceError(undefined);
      setIsEvalTraceLoading(true);

      try {
        const response = await getEvalTrace(trace.id, selectedRunId);
        const mappedTrace = mapTrace(response);
        setLoadedEvalTrace(mappedTrace);
        const flatSpans = flattenSpans(mappedTrace.spans);
        setEvalExpandedSpansIds(flatSpans.map((span) => span.id));
        setSelectedEvalSpan(flatSpans[0]);
      } catch (error) {
        setEvalTraceError(
          error instanceof Error ? error.message : "Failed to load eval trace",
        );
      } finally {
        setIsEvalTraceLoading(false);
      }
    },
    [selectedRunId],
  );

  const handleOpenGoldEpisode = useCallback(
    (goldEpisodeId: string) => {
      setSourceTab("prod");
      void handleTraceSelect({
        id: goldEpisodeId,
        name: goldEpisodeId,
        spansCount: 0,
        durationMs: 0,
        agentDescription: "",
      });
    },
    [handleTraceSelect],
  );

  const handleRefresh = useCallback(() => {
    void fetchTraceList(query);

    if (selectedTrace) {
      void handleTraceSelect(selectedTrace);
    }
  }, [fetchTraceList, handleTraceSelect, query, selectedTrace]);

  const handleEvalRefresh = useCallback(() => {
    void fetchEvalRuns();

    if (selectedRunId) {
      void fetchEvalTraceList(selectedRunId);
    }

    if (selectedRunId && selectedEvalTrace) {
      void handleEvalTraceSelect(selectedEvalTrace);
    }
  }, [
    fetchEvalRuns,
    fetchEvalTraceList,
    handleEvalTraceSelect,
    selectedEvalTrace,
    selectedRunId,
  ]);

  const selectedTraceRecord =
    loadedTrace?.traceRecord ??
    traces.find((trace) => trace.id === selectedTrace?.id) ??
    selectedTrace;

  const evalAllSpanIds = useMemo(
    () =>
      loadedEvalTrace
        ? flattenSpans(loadedEvalTrace.spans).map((span) => span.id)
        : [],
    [loadedEvalTrace],
  );

  const filteredEvalSpans = useMemo(() => {
    if (!loadedEvalTrace) return [];

    return evalSpanSearchValue.trim()
      ? filterSpansRecursively(loadedEvalTrace.spans, evalSpanSearchValue)
      : loadedEvalTrace.spans;
  }, [evalSpanSearchValue, loadedEvalTrace]);

  const selectedEvalTraceRecord =
    loadedEvalTrace?.traceRecord ??
    evalTraces.find((trace) => trace.id === selectedEvalTrace?.id) ??
    selectedEvalTrace;

  const selectedRun = evalRuns.find((run) => run.run === selectedRunId);

  const prodContent = (
    <PanelGroup direction="horizontal" className="min-h-0 flex-1">
      <Panel
        id="trace-list"
        defaultSize={28}
        minSize={traceListExpanded ? 20 : 4}
        maxSize={traceListExpanded ? 42 : 4}
        className="min-h-0 overflow-hidden"
      >
        <aside className="flex h-full min-h-0 flex-col gap-3 p-4">
          <div className="flex shrink-0 items-center gap-2">
            <TextInput
              aria-label="Search prod traces"
              className="min-w-0 flex-1"
              hideLabel
              id="trace-search"
              label="Search prod traces"
              placeholder="Search traces"
              startIcon={<Search className="size-4" />}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <Button
              aria-label="Refresh traces"
              iconStart={<RefreshCw className="size-4" />}
              onClick={handleRefresh}
              variant="secondary"
            >
              Refresh
            </Button>
          </div>

          {listError && <ErrorMessage message={listError} />}

          {isListLoading && traces.length === 0 ? (
            <StatusMessage>Loading traces...</StatusMessage>
          ) : (
            <TraceList
              traces={traces}
              expanded={traceListExpanded}
              onExpandStateChange={setTraceListExpanded}
              onTraceSelect={(trace) => void handleTraceSelect(trace)}
              selectedTrace={selectedTraceRecord}
              className="min-h-0 flex-1"
            />
          )}
        </aside>
      </Panel>

      <PanelResizeHandle className="bg-agentprism-border w-px" />

      <Panel id="trace-tree" minSize={28} className="min-h-0 overflow-hidden">
        <section className="flex h-full min-h-0 flex-col gap-3 p-4">
          {isTraceLoading ? (
            <StatusMessage>Loading trace...</StatusMessage>
          ) : traceError ? (
            <ErrorMessage message={traceError} />
          ) : loadedTrace && selectedTraceRecord ? (
            <TraceViewerTreeViewContainer
              searchValue={spanSearchValue}
              setSearchValue={setSpanSearchValue}
              handleExpandAll={() => setExpandedSpansIds(allSpanIds)}
              handleCollapseAll={() => setExpandedSpansIds([])}
              filteredSpans={filteredSpans}
              selectedSpan={selectedSpan}
              setSelectedSpan={setSelectedSpan}
              expandedSpansIds={expandedSpansIds}
              setExpandedSpansIds={setExpandedSpansIds}
              selectedTrace={selectedTraceRecord}
            />
          ) : (
            <StatusMessage>Select a trace to inspect spans.</StatusMessage>
          )}
        </section>
      </Panel>

      <PanelResizeHandle className="bg-agentprism-border w-px" />

      <Panel id="span-details" defaultSize={30} minSize={22} className="min-h-0 overflow-hidden">
        <section className="h-full min-h-0 p-4">
          {selectedSpan ? (
            <DetailsView data={selectedSpan} />
          ) : (
            <StatusMessage>Select a span to see details.</StatusMessage>
          )}
        </section>
      </Panel>
    </PanelGroup>
  );

  const evalContent = (
    <PanelGroup direction="horizontal" className="min-h-0 flex-1">
      <Panel
        id="eval-trace-list"
        defaultSize={28}
        minSize={evalTraceListExpanded ? 20 : 4}
        maxSize={evalTraceListExpanded ? 42 : 4}
        className="min-h-0 overflow-hidden"
      >
        <aside className="flex h-full min-h-0 flex-col gap-3 p-4">
          <div className="flex shrink-0 flex-col gap-2">
            <div className="flex items-center gap-2">
              <label className="sr-only" htmlFor="eval-run-select">
                Eval run
              </label>
              <select
                id="eval-run-select"
                aria-label="Eval run"
                className="border-agentprism-border bg-agentprism-background text-agentprism-foreground h-9 min-w-0 flex-1 rounded-md border px-3 text-sm"
                disabled={isRunLoading || evalRuns.length === 0}
                value={selectedRunId ?? ""}
                onChange={(event) => handleEvalRunSelect(event.target.value)}
              >
                {evalRuns.length === 0 ? (
                  <option value="">No eval runs</option>
                ) : (
                  evalRuns.map((run) => (
                    <option key={run.run} value={run.run}>
                      {formatRunOption(run)}
                    </option>
                  ))
                )}
              </select>
              <Button
                aria-label="Refresh eval runs"
                iconStart={<RefreshCw className="size-4" />}
                onClick={handleEvalRefresh}
                variant="secondary"
              >
                Refresh
              </Button>
            </div>

            {selectedRun && (
              <div className="text-agentprism-muted-foreground truncate text-xs">
                {selectedRun.executor} · {selectedRun.failedCount} failed ·{" "}
                {selectedRun.packId}
              </div>
            )}
          </div>

          {runError && <ErrorMessage message={runError} />}
          {evalListError && <ErrorMessage message={evalListError} />}

          {isRunLoading && evalRuns.length === 0 ? (
            <StatusMessage>Loading eval runs...</StatusMessage>
          ) : !selectedRunId ? (
            <StatusMessage>No eval runs found.</StatusMessage>
          ) : isEvalListLoading && evalTraces.length === 0 ? (
            <StatusMessage>Loading eval cases...</StatusMessage>
          ) : evalTraces.length === 0 ? (
            <StatusMessage>No eval cases for this run.</StatusMessage>
          ) : (
            <TraceList
              traces={evalTraces}
              expanded={evalTraceListExpanded}
              onExpandStateChange={setEvalTraceListExpanded}
              onTraceSelect={(trace) =>
                void handleEvalTraceSelect(trace as TraceRecordWithBadges)
              }
              selectedTrace={selectedEvalTraceRecord}
              className="min-h-0 flex-1"
            />
          )}
        </aside>
      </Panel>

      <PanelResizeHandle className="bg-agentprism-border w-px" />

      <Panel id="eval-trace-tree" minSize={28} className="min-h-0 overflow-hidden">
        <section className="flex h-full min-h-0 flex-col gap-3 p-4">
          {isEvalTraceLoading ? (
            <StatusMessage>Loading eval trace...</StatusMessage>
          ) : evalTraceError ? (
            <ErrorMessage message={evalTraceError} />
          ) : loadedEvalTrace && selectedEvalTraceRecord ? (
            <>
              {loadedEvalTrace.goldEpisodeId && (
                <div className="flex shrink-0 justify-end px-4">
                  <Button
                    aria-label="Open gold episode"
                    iconStart={<ExternalLink className="size-4" />}
                    onClick={() =>
                      handleOpenGoldEpisode(loadedEvalTrace.goldEpisodeId!)
                    }
                    variant="secondary"
                  >
                    open gold episode
                  </Button>
                </div>
              )}
              <TraceViewerTreeViewContainer
                searchValue={evalSpanSearchValue}
                setSearchValue={setEvalSpanSearchValue}
                handleExpandAll={() => setEvalExpandedSpansIds(evalAllSpanIds)}
                handleCollapseAll={() => setEvalExpandedSpansIds([])}
                filteredSpans={filteredEvalSpans}
                selectedSpan={selectedEvalSpan}
                setSelectedSpan={setSelectedEvalSpan}
                expandedSpansIds={evalExpandedSpansIds}
                setExpandedSpansIds={setEvalExpandedSpansIds}
                selectedTrace={selectedEvalTraceRecord}
              />
            </>
          ) : (
            <StatusMessage>Select an eval case to inspect spans.</StatusMessage>
          )}
        </section>
      </Panel>

      <PanelResizeHandle className="bg-agentprism-border w-px" />

      <Panel
        id="eval-span-details"
        defaultSize={30}
        minSize={22}
        className="min-h-0 overflow-hidden"
      >
        <section className="h-full min-h-0 p-4">
          {selectedEvalSpan ? (
            <DetailsView data={selectedEvalSpan} />
          ) : (
            <StatusMessage>Select a span to see details.</StatusMessage>
          )}
        </section>
      </Panel>
    </PanelGroup>
  );

  return (
    <main className="bg-agentprism-background text-agentprism-foreground flex h-screen min-h-0 flex-col">
      <header className="border-agentprism-border flex shrink-0 items-center justify-between gap-4 border-b px-4 py-3">
        <h1 className="text-agentprism-foreground text-base font-medium">OpenHarness Trace Viewer</h1>
        <div className="border-agentprism-border bg-agentprism-muted flex rounded-md border p-0.5">
          <button
            className={`rounded px-3 py-1.5 text-sm ${
              sourceTab === "prod"
                ? "bg-agentprism-background text-agentprism-foreground shadow-sm"
                : "text-agentprism-muted-foreground"
            }`}
            type="button"
            onClick={() => setSourceTab("prod")}
          >
            prod
          </button>
          <button
            className={`rounded px-3 py-1.5 text-sm ${
              sourceTab === "eval"
                ? "bg-agentprism-background text-agentprism-foreground shadow-sm"
                : "text-agentprism-muted-foreground"
            }`}
            type="button"
            onClick={() => setSourceTab("eval")}
          >
            прокачки
          </button>
        </div>
      </header>

      {sourceTab === "prod" ? prodContent : evalContent}
    </main>
  );
}

function mapEvalTraceSummary(dto: EvalTraceSummaryDTO): TraceRecordWithBadges {
  const badges: BadgeProps[] = [
    {
      label: <EvalStatusLabel status={dto.status} />,
    },
  ];
  const score = formatEvalScore(dto.score);

  if (score) {
    badges.push({ label: `score ${score}` });
  }

  badges.push({ label: `${dto.passCount}/${dto.sampleCount} passed` });

  return {
    id: dto.id,
    name: dto.name,
    spansCount: dto.spansCount,
    durationMs: 0,
    agentDescription: dto.goldEpisodeId,
    badges,
  };
}

function EvalStatusLabel({ status }: { status: EvalTraceSummaryDTO["status"] }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1">
      <span
        className={`block size-1.5 shrink-0 rounded-full ${EVAL_STATUS_DOT_CLASSES[status]}`}
      />
      <span className="truncate">{status}</span>
    </span>
  );
}

const EVAL_STATUS_DOT_CLASSES: Record<EvalTraceSummaryDTO["status"], string> = {
  success: "bg-agentprism-success",
  error: "bg-agentprism-error",
  warning: "bg-agentprism-warning",
};

function formatRunOption(run: EvalRunDTO): string {
  return `${run.scorer} · ${formatDateTime(run.createdAt)} · ${run.passedCount}/${run.caseCount}`;
}

function formatDateTime(epochMs: number): string {
  const date = new Date(epochMs);

  if (Number.isNaN(date.getTime())) {
    return "unknown date";
  }

  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatEvalScore(score: number | null): string | undefined {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return undefined;
  }

  if (Number.isInteger(score)) {
    return String(score);
  }

  return score.toFixed(3).replace(/\.?0+$/, "");
}

function StatusMessage({ children }: { children: string }) {
  return (
    <div className="border-agentprism-border bg-agentprism-background flex h-full min-h-40 items-center justify-center rounded-md border p-6 text-sm text-agentprism-muted-foreground">
      {children}
    </div>
  );
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="border-agentprism-destructive/30 bg-agentprism-destructive/10 text-agentprism-destructive flex items-center gap-2 rounded-md border p-3 text-sm">
      <AlertCircle className="size-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export default App;
