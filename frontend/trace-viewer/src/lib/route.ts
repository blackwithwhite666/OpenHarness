// Flat hash-based routing so every view has a direct, shareable URL.
// Example: #tab=eval&run=eval_report_x.json&case=case-1&sample=2&chat=gold

export type ChatKind = "session" | "observed" | "gold";

export interface AppRoute {
  tab: "prod" | "eval";
  trace?: string;
  run?: string;
  caseId?: string;
  sample?: number;
  chat?: ChatKind;
}

function isChatKind(value: string | null): value is ChatKind {
  return value === "session" || value === "observed" || value === "gold";
}

export function parseHash(hash: string): AppRoute {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const params = new URLSearchParams(raw);
  const tab = params.get("tab") === "eval" ? "eval" : "prod";
  const sampleRaw = params.get("sample");
  const sample =
    sampleRaw != null && /^\d+$/.test(sampleRaw) ? Number(sampleRaw) : undefined;
  const chatRaw = params.get("chat");

  return {
    tab,
    trace: params.get("trace") || undefined,
    run: params.get("run") || undefined,
    caseId: params.get("case") || undefined,
    sample,
    chat: isChatKind(chatRaw) ? chatRaw : undefined,
  };
}

export function serializeRoute(route: AppRoute): string {
  const params = new URLSearchParams();
  params.set("tab", route.tab);

  if (route.tab === "prod") {
    if (route.trace) {
      params.set("trace", route.trace);
    }
    if (route.chat === "session") {
      params.set("chat", "session");
    }
  } else {
    if (route.run) {
      params.set("run", route.run);
    }
    if (route.caseId) {
      params.set("case", route.caseId);
      if (typeof route.sample === "number" && route.sample > 0) {
        params.set("sample", String(route.sample));
      }
    }
    if (route.chat === "observed" || route.chat === "gold") {
      params.set("chat", route.chat);
    }
  }

  return `#${params.toString()}`;
}
