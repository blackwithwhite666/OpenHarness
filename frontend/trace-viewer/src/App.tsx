import type { TraceRecord, TraceSpan } from "@evilmartians/agent-prism-types";

import { flattenSpans, filterSpansRecursively } from "@evilmartians/agent-prism-data";
import {
  AlertCircle,
  ExternalLink,
  GitCompareArrows,
  RefreshCw,
  Search,
} from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
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
  type TraceSummaryDTO,
} from "./lib/api";
import { mapTrace, mapTraceSummary } from "./lib/mapTrace";

type SourceTab = "prod" | "eval";
type StatusFilter = "all" | "passed" | "failed";

type TraceRecordWithBadges = TraceRecord & {
  badges?: BadgeProps[];
};

type ProdTraceRecord = TraceRecord & {
  status: TraceSummaryDTO["status"];
};

type EvalTraceRecord = TraceRecordWithBadges & {
  status: EvalTraceSummaryDTO["status"];
  goldEpisodeId: string;
  sampleCount: number;
};

interface LoadedTrace {
  traceRecord: TraceRecordWithBadges;
  spans: TraceSpan[];
  goldEpisodeId?: string;
  badges?: BadgeProps[];
  sample?: number;
  sampleCount?: number;
}

function App() {
  const [sourceTab, setSourceTab] = useState<SourceTab>("prod");
  const [query, setQuery] = useState("");
  const [prodStatusFilter, setProdStatusFilter] = useState<StatusFilter>("all");
  const [traces, setTraces] = useState<ProdTraceRecord[]>([]);
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
  const [evalTraces, setEvalTraces] = useState<EvalTraceRecord[]>([]);
  const [selectedEvalTrace, setSelectedEvalTrace] =
    useState<EvalTraceRecord | undefined>();
  const [selectedEvalSample, setSelectedEvalSample] = useState(0);
  const [evalQuery, setEvalQuery] = useState("");
  const [evalStatusFilter, setEvalStatusFilter] = useState<StatusFilter>("all");
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
  const [isGoldCompareEnabled, setIsGoldCompareEnabled] = useState(false);
  const [loadedGoldTrace, setLoadedGoldTrace] = useState<LoadedTrace | undefined>();
  const [selectedGoldSpan, setSelectedGoldSpan] = useState<TraceSpan | undefined>();
  const [goldExpandedSpansIds, setGoldExpandedSpansIds] = useState<string[]>([]);
  const [goldSpanSearchValue, setGoldSpanSearchValue] = useState("");
  const [isGoldTraceLoading, setIsGoldTraceLoading] = useState(false);
  const [goldTraceError, setGoldTraceError] = useState<string | undefined>();

  const fetchTraceList = useCallback(async (search: string) => {
    setIsListLoading(true);
    setListError(undefined);

    try {
      const response = await listTraces(search);
      const nextTraces = response.traces.map((trace) => ({
        ...mapTraceSummary(trace),
        status: trace.status,
      }));
      setTraces(nextTraces);
    } catch (error) {
      setListError(error instanceof Error ? error.message : "Failed to load traces");
    } finally {
      setIsListLoading(false);
    }
  }, []);

  const clearGoldTraceState = useCallback(() => {
    setLoadedGoldTrace(undefined);
    setSelectedGoldSpan(undefined);
    setGoldSpanSearchValue("");
    setGoldExpandedSpansIds([]);
    setGoldTraceError(undefined);
    setIsGoldTraceLoading(false);
  }, []);

  const clearEvalTraceSelection = useCallback(() => {
    setSelectedEvalTrace(undefined);
    setSelectedEvalSample(0);
    setLoadedEvalTrace(undefined);
    setSelectedEvalSpan(undefined);
    setEvalSpanSearchValue("");
    setEvalExpandedSpansIds([]);
    setEvalTraceError(undefined);
    setIsGoldCompareEnabled(false);
    clearGoldTraceState();
  }, [clearGoldTraceState]);

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

  const loadEvalTrace = useCallback(
    async (trace: EvalTraceRecord, sample: number) => {
      if (!selectedRunId) return;

      setSelectedEvalTrace(trace);
      setSelectedEvalSample(sample);
      setLoadedEvalTrace(undefined);
      setSelectedEvalSpan(undefined);
      setEvalSpanSearchValue("");
      setEvalExpandedSpansIds([]);
      setEvalTraceError(undefined);
      clearGoldTraceState();
      setIsEvalTraceLoading(true);

      try {
        const response = await getEvalTrace(trace.id, selectedRunId, sample);
        const mappedTrace = mapTrace(response);
        setLoadedEvalTrace(mappedTrace);
        setSelectedEvalSample(mappedTrace.sample ?? sample);
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
    [clearGoldTraceState, selectedRunId],
  );

  const handleEvalTraceSelect = useCallback(
    async (trace: EvalTraceRecord) => {
      await loadEvalTrace(trace, 0);
    },
    [loadEvalTrace],
  );

  const handleEvalSampleSelect = useCallback(
    async (sample: number) => {
      if (!selectedEvalTrace) return;

      await loadEvalTrace(selectedEvalTrace, sample);
    },
    [loadEvalTrace, selectedEvalTrace],
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
      void loadEvalTrace(selectedEvalTrace, selectedEvalSample);
    }
  }, [
    fetchEvalRuns,
    fetchEvalTraceList,
    loadEvalTrace,
    selectedEvalTrace,
    selectedEvalSample,
    selectedRunId,
  ]);

  const selectedTraceRecord =
    loadedTrace?.traceRecord ??
    traces.find((trace) => trace.id === selectedTrace?.id) ??
    selectedTrace;

  const filteredTraces = useMemo(
    () =>
      traces.filter((trace) => matchesStatusFilter(trace.status, prodStatusFilter)),
    [prodStatusFilter, traces],
  );

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

  useEffect(() => {
    if (!isGoldCompareEnabled) return;

    const goldEpisodeId = loadedEvalTrace?.goldEpisodeId;

    if (!goldEpisodeId) {
      setIsGoldCompareEnabled(false);
      clearGoldTraceState();
      return;
    }

    let ignore = false;

    setLoadedGoldTrace(undefined);
    setSelectedGoldSpan(undefined);
    setGoldSpanSearchValue("");
    setGoldExpandedSpansIds([]);
    setGoldTraceError(undefined);
    setIsGoldTraceLoading(true);

    getTrace(goldEpisodeId)
      .then((response) => {
        if (ignore) return;

        const mappedTrace = mapTrace(response);
        setLoadedGoldTrace(mappedTrace);

        const flatSpans = flattenSpans(mappedTrace.spans);
        setGoldExpandedSpansIds(flatSpans.map((span) => span.id));
        setSelectedGoldSpan(flatSpans[0]);
      })
      .catch((error) => {
        if (ignore) return;

        setGoldTraceError(
          error instanceof Error ? error.message : "Failed to load gold episode",
        );
      })
      .finally(() => {
        if (!ignore) {
          setIsGoldTraceLoading(false);
        }
      });

    return () => {
      ignore = true;
    };
  }, [clearGoldTraceState, isGoldCompareEnabled, loadedEvalTrace?.goldEpisodeId]);

  const goldAllSpanIds = useMemo(
    () =>
      loadedGoldTrace
        ? flattenSpans(loadedGoldTrace.spans).map((span) => span.id)
        : [],
    [loadedGoldTrace],
  );

  const filteredGoldSpans = useMemo(() => {
    if (!loadedGoldTrace) return [];

    return goldSpanSearchValue.trim()
      ? filterSpansRecursively(loadedGoldTrace.spans, goldSpanSearchValue)
      : loadedGoldTrace.spans;
  }, [goldSpanSearchValue, loadedGoldTrace]);

  const selectedEvalTraceRecord =
    loadedEvalTrace?.traceRecord ??
    evalTraces.find((trace) => trace.id === selectedEvalTrace?.id) ??
    selectedEvalTrace;
  const selectedGoldTraceRecord = loadedGoldTrace?.traceRecord;
  const isGoldCompareActive =
    isGoldCompareEnabled && Boolean(loadedEvalTrace?.goldEpisodeId);
  const trajectoryComparison = useMemo(
    () =>
      loadedEvalTrace && loadedGoldTrace
        ? compareTrajectories(loadedEvalTrace.spans, loadedGoldTrace.spans)
        : undefined,
    [loadedEvalTrace, loadedGoldTrace],
  );

  const selectedRun = evalRuns.find((run) => run.run === selectedRunId);
  const selectedEvalSampleCount =
    loadedEvalTrace?.sampleCount ?? selectedEvalTrace?.sampleCount ?? 0;
  const selectedEvalSampleStatus = loadedEvalTrace?.spans[0]?.status;
  const filteredEvalTraces = useMemo(() => {
    const normalizedQuery = evalQuery.trim().toLocaleLowerCase();

    return evalTraces.filter((trace) => {
      if (!matchesStatusFilter(trace.status, evalStatusFilter)) {
        return false;
      }

      if (!normalizedQuery) {
        return true;
      }

      return [trace.id, trace.goldEpisodeId].some((value) =>
        value.toLocaleLowerCase().includes(normalizedQuery),
      );
    });
  }, [evalQuery, evalStatusFilter, evalTraces]);

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
          <StatusFilterControl
            ariaLabel="Filter prod traces by status"
            value={prodStatusFilter}
            onChange={setProdStatusFilter}
          />

          {listError && <ErrorMessage message={listError} />}

          {isListLoading && traces.length === 0 ? (
            <StatusMessage>Loading traces...</StatusMessage>
          ) : traces.length > 0 && filteredTraces.length === 0 ? (
            <StatusMessage>No traces match the filters.</StatusMessage>
          ) : (
            <TraceList
              traces={filteredTraces}
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

            <div className="flex items-center gap-2">
              <TextInput
                aria-label="Search eval cases"
                className="min-w-0 flex-1"
                hideLabel
                id="eval-case-search"
                label="Search eval cases"
                placeholder="Search cases"
                startIcon={<Search className="size-4" />}
                value={evalQuery}
                onChange={(event) => setEvalQuery(event.target.value)}
              />
              <StatusFilterControl
                ariaLabel="Filter eval cases by status"
                value={evalStatusFilter}
                onChange={setEvalStatusFilter}
              />
            </div>
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
          ) : filteredEvalTraces.length === 0 ? (
            <StatusMessage>No eval cases match the filters.</StatusMessage>
          ) : (
            <TraceList
              traces={filteredEvalTraces}
              expanded={evalTraceListExpanded}
              onExpandStateChange={setEvalTraceListExpanded}
              onTraceSelect={(trace) => void handleEvalTraceSelect(trace as EvalTraceRecord)}
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
              {selectedEvalSampleCount > 1 && (
                <SampleSelector
                  count={selectedEvalSampleCount}
                  selectedSample={selectedEvalSample}
                  selectedSampleStatus={selectedEvalSampleStatus}
                  onSelect={(sample) => void handleEvalSampleSelect(sample)}
                />
              )}
              <div className="flex shrink-0 flex-wrap justify-end gap-2 px-4">
                <Button
                  aria-label="Compare to gold"
                  aria-pressed={isGoldCompareActive}
                  disabled={!loadedEvalTrace.goldEpisodeId}
                  iconStart={<GitCompareArrows className="size-4" />}
                  onClick={() => setIsGoldCompareEnabled((enabled) => !enabled)}
                  variant={isGoldCompareActive ? "primary" : "secondary"}
                >
                  Compare to gold
                </Button>
                {loadedEvalTrace.goldEpisodeId && (
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
                )}
              </div>
              {isGoldCompareActive ? (
                <div className="flex min-h-0 flex-1 flex-col gap-3">
                  <TrajectoryComparisonNote comparison={trajectoryComparison} />
                  <div className="grid min-h-0 flex-1 grid-cols-2 gap-3">
                    <CompareTraceColumn
                      title="Observed (eval)"
                      subtitle={selectedEvalTraceRecord.name}
                    >
                      <TraceViewerTreeViewContainer
                        searchValue={evalSpanSearchValue}
                        setSearchValue={setEvalSpanSearchValue}
                        handleExpandAll={() =>
                          setEvalExpandedSpansIds(evalAllSpanIds)
                        }
                        handleCollapseAll={() => setEvalExpandedSpansIds([])}
                        filteredSpans={filteredEvalSpans}
                        selectedSpan={selectedEvalSpan}
                        setSelectedSpan={setSelectedEvalSpan}
                        expandedSpansIds={evalExpandedSpansIds}
                        setExpandedSpansIds={setEvalExpandedSpansIds}
                        selectedTrace={selectedEvalTraceRecord}
                        showHeader={false}
                      />
                    </CompareTraceColumn>

                    <CompareTraceColumn
                      title="Gold episode"
                      subtitle={loadedEvalTrace.goldEpisodeId}
                    >
                      {isGoldTraceLoading ? (
                        <StatusMessage>Loading gold episode...</StatusMessage>
                      ) : goldTraceError ? (
                        <ErrorMessage message={goldTraceError} />
                      ) : loadedGoldTrace && selectedGoldTraceRecord ? (
                        <TraceViewerTreeViewContainer
                          searchValue={goldSpanSearchValue}
                          setSearchValue={setGoldSpanSearchValue}
                          handleExpandAll={() =>
                            setGoldExpandedSpansIds(goldAllSpanIds)
                          }
                          handleCollapseAll={() => setGoldExpandedSpansIds([])}
                          filteredSpans={filteredGoldSpans}
                          selectedSpan={selectedGoldSpan}
                          setSelectedSpan={setSelectedGoldSpan}
                          expandedSpansIds={goldExpandedSpansIds}
                          setExpandedSpansIds={setGoldExpandedSpansIds}
                          selectedTrace={selectedGoldTraceRecord}
                          showHeader={false}
                        />
                      ) : (
                        <StatusMessage>Gold episode is not loaded.</StatusMessage>
                      )}
                    </CompareTraceColumn>
                  </div>
                </div>
              ) : (
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
              )}
            </>
          ) : (
            <StatusMessage>Select an eval case to inspect spans.</StatusMessage>
          )}
        </section>
      </Panel>

      {!isGoldCompareActive && (
        <>
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
        </>
      )}
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

function StatusFilterControl({
  ariaLabel,
  value,
  onChange,
}: {
  ariaLabel: string;
  value: StatusFilter;
  onChange: (value: StatusFilter) => void;
}) {
  const filters: StatusFilter[] = ["all", "passed", "failed"];

  return (
    <div
      aria-label={ariaLabel}
      className="border-agentprism-border bg-agentprism-muted flex shrink-0 rounded-md border p-0.5"
      role="group"
    >
      {filters.map((filter) => (
        <button
          key={filter}
          aria-pressed={value === filter}
          className={`rounded px-2.5 py-1 text-xs ${
            value === filter
              ? "bg-agentprism-background text-agentprism-foreground shadow-sm"
              : "text-agentprism-muted-foreground"
          }`}
          type="button"
          onClick={() => onChange(filter)}
        >
          {filter}
        </button>
      ))}
    </div>
  );
}

function SampleSelector({
  count,
  selectedSample,
  selectedSampleStatus,
  onSelect,
}: {
  count: number;
  selectedSample: number;
  selectedSampleStatus?: TraceSpan["status"];
  onSelect: (sample: number) => void;
}) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-1 px-4">
      {Array.from({ length: count }, (_, sample) => {
        const isSelected = sample === selectedSample;

        return (
          <button
            key={sample}
            aria-pressed={isSelected}
            className={`border-agentprism-border inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs ${
              isSelected
                ? "bg-agentprism-secondary text-agentprism-foreground"
                : "bg-agentprism-background text-agentprism-muted-foreground hover:bg-agentprism-secondary/45"
            }`}
            type="button"
            onClick={() => onSelect(sample)}
          >
            {isSelected && selectedSampleStatus && (
              <span
                className={`block size-1.5 shrink-0 rounded-full ${TRACE_STATUS_DOT_CLASSES[selectedSampleStatus]}`}
              />
            )}
            #{sample}
          </button>
        );
      })}
    </div>
  );
}

type TrajectoryComparisonKind =
  | "match"
  | "diverge"
  | "observed-prefix"
  | "gold-prefix";

interface TrajectoryComparison {
  kind: TrajectoryComparisonKind;
  observedLength: number;
  goldLength: number;
  step?: number;
  observedTitle?: string;
  goldTitle?: string;
}

function CompareTraceColumn({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <div className="border-agentprism-border bg-agentprism-background flex min-h-0 min-w-0 flex-col overflow-hidden rounded-md border">
      <div className="border-agentprism-border flex shrink-0 flex-col gap-0.5 border-b px-4 py-3">
        <h2 className="text-agentprism-foreground text-sm font-medium">{title}</h2>
        {subtitle && (
          <p
            className="text-agentprism-muted-foreground truncate text-xs"
            title={subtitle}
          >
            {subtitle}
          </p>
        )}
      </div>
      <div className="flex min-h-0 flex-1 flex-col p-3">{children}</div>
    </div>
  );
}

function TrajectoryComparisonNote({
  comparison,
}: {
  comparison?: TrajectoryComparison;
}) {
  if (!comparison) return null;

  const isMatch = comparison.kind === "match";

  return (
    <div
      className={`shrink-0 rounded-md border px-4 py-2 text-sm ${
        isMatch
          ? "border-agentprism-success/30 bg-agentprism-success-muted text-agentprism-success-muted-foreground"
          : "border-agentprism-warning/30 bg-agentprism-warning-muted text-agentprism-warning-muted-foreground"
      }`}
    >
      <TrajectoryComparisonText comparison={comparison} />
    </div>
  );
}

function TrajectoryComparisonText({
  comparison,
}: {
  comparison: TrajectoryComparison;
}) {
  if (comparison.kind === "match") {
    return <>trajectories match ({comparison.observedLength} steps)</>;
  }

  if (comparison.kind === "observed-prefix") {
    return (
      <>
        observed is a prefix of gold; next gold step {comparison.step}:{" "}
        <TrajectoryStep>{comparison.goldTitle}</TrajectoryStep>
      </>
    );
  }

  if (comparison.kind === "gold-prefix") {
    return (
      <>
        gold is a prefix of observed; next observed step {comparison.step}:{" "}
        <TrajectoryStep>{comparison.observedTitle}</TrajectoryStep>
      </>
    );
  }

  return (
    <>
      diverges at step {comparison.step}: observed{" "}
      <TrajectoryStep>{comparison.observedTitle}</TrajectoryStep> vs gold{" "}
      <TrajectoryStep>{comparison.goldTitle}</TrajectoryStep>
    </>
  );
}

function TrajectoryStep({ children }: { children?: string }) {
  return (
    <code className="bg-agentprism-background/70 text-agentprism-foreground rounded px-1 py-0.5 text-xs">
      {children || "(missing)"}
    </code>
  );
}

function compareTrajectories(
  observedSpans: TraceSpan[],
  goldSpans: TraceSpan[],
): TrajectoryComparison {
  const observedSequence = getTrajectorySequence(observedSpans);
  const goldSequence = getTrajectorySequence(goldSpans);
  const sharedLength = Math.min(observedSequence.length, goldSequence.length);

  for (let index = 0; index < sharedLength; index += 1) {
    if (observedSequence[index] !== goldSequence[index]) {
      return {
        kind: "diverge",
        observedLength: observedSequence.length,
        goldLength: goldSequence.length,
        step: index + 1,
        observedTitle: observedSequence[index],
        goldTitle: goldSequence[index],
      };
    }
  }

  if (observedSequence.length === goldSequence.length) {
    return {
      kind: "match",
      observedLength: observedSequence.length,
      goldLength: goldSequence.length,
    };
  }

  if (observedSequence.length < goldSequence.length) {
    return {
      kind: "observed-prefix",
      observedLength: observedSequence.length,
      goldLength: goldSequence.length,
      step: observedSequence.length + 1,
      goldTitle: goldSequence[observedSequence.length],
    };
  }

  return {
    kind: "gold-prefix",
    observedLength: observedSequence.length,
    goldLength: goldSequence.length,
    step: goldSequence.length + 1,
    observedTitle: observedSequence[goldSequence.length],
  };
}

function getTrajectorySequence(spans: TraceSpan[]): string[] {
  const firstRoot = spans[0];
  const rootChildren = firstRoot?.children ?? [];
  const trajectorySpans =
    spans.length === 1 && rootChildren.length > 0 ? rootChildren : spans;

  return trajectorySpans.map((span) => span.title.trim() || "(untitled)");
}

function matchesStatusFilter(
  status: TraceSummaryDTO["status"] | EvalTraceSummaryDTO["status"],
  filter: StatusFilter,
): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "passed") {
    return status === "success";
  }
  return status !== "success";
}

function mapEvalTraceSummary(dto: EvalTraceSummaryDTO): EvalTraceRecord {
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
    status: dto.status,
    goldEpisodeId: dto.goldEpisodeId,
    sampleCount: dto.sampleCount,
  };
}

function EvalStatusLabel({ status }: { status: EvalTraceSummaryDTO["status"] }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1">
      <span
        className={`block size-1.5 shrink-0 rounded-full ${TRACE_STATUS_DOT_CLASSES[status]}`}
      />
      <span className="truncate">{status}</span>
    </span>
  );
}

const TRACE_STATUS_DOT_CLASSES: Record<TraceSpan["status"], string> = {
  success: "bg-agentprism-success",
  error: "bg-agentprism-error",
  warning: "bg-agentprism-warning",
  pending: "bg-agentprism-pending",
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
