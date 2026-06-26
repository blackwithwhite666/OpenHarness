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
        className="border-agentprism-border bg-agentprism-background flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="border-agentprism-border flex shrink-0 items-center gap-3 border-b px-4 py-3">
          <MessagesSquare className="text-agentprism-muted-foreground size-4 shrink-0" />
          <div className="min-w-0 flex-1">
            <h2 className="text-agentprism-foreground truncate text-sm font-medium" title={title}>
              {title}
            </h2>
            {data && (
              <p className="text-agentprism-muted-foreground truncate text-xs">
                {data.kind === "observed" ? "observed dialog" : "chat session"}
                {data.sessionId ? ` · ${data.sessionId}` : ""}
                {` · ${data.messages.length} messages`}
              </p>
            )}
          </div>
          {shareUrl && (
            <button
              type="button"
              className="text-agentprism-muted-foreground hover:text-agentprism-foreground text-xs underline"
              onClick={() => void navigator.clipboard?.writeText(shareUrl)}
              title={shareUrl}
            >
              copy link
            </button>
          )}
          <button
            type="button"
            aria-label="Close conversation"
            className="text-agentprism-muted-foreground hover:text-agentprism-foreground"
            onClick={onClose}
          >
            <X className="size-4" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {loading ? (
            <ChatStatus>Loading conversation…</ChatStatus>
          ) : error ? (
            <div className="border-agentprism-destructive/30 bg-agentprism-destructive/10 text-agentprism-destructive rounded-md border p-3 text-sm">
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
  const isAnchor =
    !!anchorEpisodeId && message.episodeId === anchorEpisodeId;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`flex max-w-[85%] flex-col gap-1.5 rounded-lg border px-3 py-2 ${
          isUser
            ? "border-agentprism-border bg-agentprism-secondary/50"
            : "border-agentprism-border bg-agentprism-muted"
        } ${isAnchor ? "ring-agentprism-warning/60 ring-2" : ""}`}
      >
        <div className="text-agentprism-muted-foreground flex items-center gap-2 text-[11px] uppercase tracking-wide">
          <span>{isUser ? "user" : "assistant"}</span>
          {message.status === "error" && (
            <span className="text-agentprism-destructive">· tool error</span>
          )}
          {message.episodeId && onOpenEpisode && (
            <button
              type="button"
              className="hover:text-agentprism-foreground inline-flex items-center gap-0.5 normal-case underline"
              onClick={() => onOpenEpisode(message.episodeId!)}
              title={`Open trace ${message.episodeId}`}
            >
              <ExternalLink className="size-3" />
              trace
            </button>
          )}
        </div>

        {message.text ? (
          <p className="text-agentprism-foreground whitespace-pre-wrap break-words text-sm">
            {message.text}
          </p>
        ) : (
          <p className="text-agentprism-muted-foreground text-sm italic">
            (no text)
          </p>
        )}

        {message.toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-0.5">
            {message.toolCalls.map((tool, index) => (
              <span
                key={index}
                className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] ${
                  tool.status === "error"
                    ? "border-agentprism-destructive/40 text-agentprism-destructive"
                    : "border-agentprism-border text-agentprism-muted-foreground"
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
          ? "border-agentprism-success/30 bg-agentprism-success-muted text-agentprism-success-muted-foreground"
          : "border-agentprism-warning/30 bg-agentprism-warning-muted text-agentprism-warning-muted-foreground"
      }`}
    >
      <span className="font-medium">judge: {verdict}</span>
      {reason && <span> — {reason}</span>}
    </div>
  );
}

function ChatStatus({ children }: { children: string }) {
  return (
    <div className="text-agentprism-muted-foreground flex h-40 items-center justify-center text-sm">
      {children}
    </div>
  );
}
