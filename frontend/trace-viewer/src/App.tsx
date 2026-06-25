import type { TraceRecord, TraceSpan } from "@evilmartians/agent-prism-types";

import { flattenSpans, filterSpansRecursively } from "@evilmartians/agent-prism-data";
import { AlertCircle, RefreshCw, Search } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";

import { Button } from "./components/agent-prism/Button";
import { DetailsView } from "./components/agent-prism/DetailsView/DetailsView";
import { TextInput } from "./components/agent-prism/TextInput";
import { TraceList } from "./components/agent-prism/TraceList/TraceList";
import { TraceViewerTreeViewContainer } from "./components/agent-prism/TraceViewer/TraceViewerTreeViewContainer";
import { getTrace, listTraces } from "./lib/api";
import { mapTrace, mapTraceSummary } from "./lib/mapTrace";

type SourceTab = "prod" | "eval";

interface LoadedTrace {
  traceRecord: TraceRecord;
  spans: TraceSpan[];
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

  useEffect(() => {
    if (sourceTab !== "prod") return;

    const handle = window.setTimeout(() => {
      void fetchTraceList(query);
    }, 250);

    return () => window.clearTimeout(handle);
  }, [fetchTraceList, query, sourceTab]);

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

  const handleRefresh = useCallback(() => {
    void fetchTraceList(query);

    if (selectedTrace) {
      void handleTraceSelect(selectedTrace);
    }
  }, [fetchTraceList, handleTraceSelect, query, selectedTrace]);

  const selectedTraceRecord =
    loadedTrace?.traceRecord ??
    traces.find((trace) => trace.id === selectedTrace?.id) ??
    selectedTrace;

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

      {sourceTab === "prod" ? (
        prodContent
      ) : (
        <section className="flex flex-1 items-center justify-center p-8">
          <div className="border-agentprism-border bg-agentprism-background rounded-md border p-6 text-sm text-agentprism-muted-foreground">
            coming in P1
          </div>
        </section>
      )}
    </main>
  );
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
