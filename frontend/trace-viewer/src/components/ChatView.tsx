import { ExternalLink, MessagesSquare, X } from "lucide-react";

import type { ConversationDTO, ConversationMessageDTO } from "../lib/api";

interface ChatViewProps {
  title: string;
  data?: ConversationDTO;
  loading: boolean;
  error?: string;
  onClose: () => void;
  onOpenEpisode?: (episodeId: string) => void;
  shareUrl?: string;
}

// Uses concrete Tailwind colors (not the agentprism-* tokens) because those
// tokens are CSS variables that were never wired into Tailwind utilities, so
// `bg-agentprism-*` produces no rule. A modal sitting over a dark backdrop
// needs a real, opaque background to stay readable.
export function ChatView({
  title,
  data,
  loading,
  error,
  onClose,
  onOpenEpisode,
  shareUrl,
}: ChatViewProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Conversation transcript"
      onClick={onClose}
    >
      <div
        className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-neutral-200 bg-white text-neutral-900 shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-neutral-200 px-4 py-3">
          <MessagesSquare className="size-4 shrink-0 text-neutral-400" />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-sm font-medium text-neutral-900" title={title}>
              {title}
            </h2>
            {data && (
              <p className="truncate text-xs text-neutral-500">
                {data.kind === "observed" ? "observed dialog" : "chat session"}
                {data.sessionId ? ` · ${data.sessionId}` : ""}
                {` · ${data.messages.length} messages`}
              </p>
            )}
          </div>
          {shareUrl && (
            <button
              type="button"
              className="text-xs text-neutral-500 underline hover:text-neutral-900"
              onClick={() => void navigator.clipboard?.writeText(shareUrl)}
              title={shareUrl}
            >
              copy link
            </button>
          )}
          <button
            type="button"
            aria-label="Close conversation"
            className="text-neutral-500 hover:text-neutral-900"
            onClick={onClose}
          >
            <X className="size-4" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto bg-neutral-50 p-4">
          {loading ? (
            <ChatStatus>Loading conversation…</ChatStatus>
          ) : error ? (
            <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          ) : !data || data.messages.length === 0 ? (
            <ChatStatus>No messages in this conversation.</ChatStatus>
          ) : (
            <div className="flex flex-col gap-3">
              {data.kind === "observed" && data.judgeVerdict && (
                <JudgeNote verdict={data.judgeVerdict} reason={data.judgeReason} />
              )}
              {data.messages.map((message, index) => (
                <ChatBubble
                  key={index}
                  message={message}
                  anchorEpisodeId={data.anchorEpisodeId}
                  onOpenEpisode={onOpenEpisode}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ChatBubble({
  message,
  anchorEpisodeId,
  onOpenEpisode,
}: {
  message: ConversationMessageDTO;
  anchorEpisodeId?: string;
  onOpenEpisode?: (episodeId: string) => void;
}) {
  const isUser = message.role === "user";
  const isAnchor = !!anchorEpisodeId && message.episodeId === anchorEpisodeId;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`flex max-w-[85%] flex-col gap-1.5 rounded-lg border px-3 py-2 ${
          isUser
            ? "border-blue-200 bg-blue-50"
            : "border-neutral-200 bg-white"
        } ${isAnchor ? "ring-2 ring-amber-400" : ""}`}
      >
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-neutral-500">
          <span>{isUser ? "user" : "assistant"}</span>
          {message.status === "error" && (
            <span className="text-red-600">· tool error</span>
          )}
          {message.episodeId && onOpenEpisode && (
            <button
              type="button"
              className="inline-flex items-center gap-0.5 normal-case underline hover:text-neutral-900"
              onClick={() => onOpenEpisode(message.episodeId!)}
              title={`Open trace ${message.episodeId}`}
            >
              <ExternalLink className="size-3" />
              trace
            </button>
          )}
        </div>

        {message.text ? (
          <p className="whitespace-pre-wrap break-words text-sm text-neutral-900">
            {message.text}
          </p>
        ) : (
          <p className="text-sm italic text-neutral-400">(no text)</p>
        )}

        {message.toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-0.5">
            {message.toolCalls.map((tool, index) => (
              <span
                key={index}
                className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] ${
                  tool.status === "error"
                    ? "border-red-300 text-red-600"
                    : "border-neutral-300 text-neutral-600"
                }`}
                title={tool.name}
              >
                {tool.name}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function JudgeNote({
  verdict,
  reason,
}: {
  verdict: string;
  reason?: string | null;
}) {
  const isPass = verdict.toLowerCase().startsWith("pass");

  return (
    <div
      className={`rounded-md border px-3 py-2 text-sm ${
        isPass
          ? "border-emerald-200 bg-emerald-50 text-emerald-800"
          : "border-amber-200 bg-amber-50 text-amber-800"
      }`}
    >
      <span className="font-medium">judge: {verdict}</span>
      {reason && <span> — {reason}</span>}
    </div>
  );
}

function ChatStatus({ children }: { children: string }) {
  return (
    <div className="flex h-40 items-center justify-center text-sm text-neutral-500">
      {children}
    </div>
  );
}
