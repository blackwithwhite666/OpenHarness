import { Fragment, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Download,
  FileAudio,
  FileImage,
  FileVideo,
  Paperclip,
} from "lucide-react";
import { lexer, type Token, type Tokens } from "marked";

type AttachmentKind =
  | "attach"
  | "audio"
  | "file"
  | "image"
  | "photo"
  | "video"
  | "voice";

type AttachmentDisplayKind = "audio" | "file" | "image" | "video";

interface AttachmentMarker {
  kind: AttachmentKind;
  path: string;
}

type MessageSegment =
  | { type: "markdown"; text: string }
  | { type: "attachment"; attachment: AttachmentMarker };

const MEDIA_MARKER_RE =
  /\[\[attach:\s*([^\]\r\n]+?)\s*\]\]|\[(voice|audio|video|photo|image|file):\s*([^\]\r\n]+?)\s*\]/gi;

const IMAGE_SUFFIXES = new Set([
  ".apng",
  ".avif",
  ".bmp",
  ".gif",
  ".jpeg",
  ".jpg",
  ".png",
  ".webp",
]);
const AUDIO_SUFFIXES = new Set([
  ".aac",
  ".flac",
  ".m4a",
  ".mp3",
  ".oga",
  ".ogg",
  ".opus",
  ".wav",
]);
const VIDEO_SUFFIXES = new Set([".m4v", ".mov", ".mp4", ".ogv", ".webm"]);

export function MarkdownMessage({ text }: { text: string }) {
  const segments = useMemo(() => splitMessageText(text), [text]);

  return (
    <div className="flex flex-col gap-2 break-words text-sm text-neutral-900">
      {segments.map((segment, index) =>
        segment.type === "markdown" ? (
          <MarkdownBlocks key={index} content={segment.text} />
        ) : (
          <AttachmentPreview key={index} attachment={segment.attachment} />
        ),
      )}
    </div>
  );
}

function splitMessageText(text: string): MessageSegment[] {
  const segments: MessageSegment[] = [];
  let lastIndex = 0;
  MEDIA_MARKER_RE.lastIndex = 0;

  for (const match of text.matchAll(MEDIA_MARKER_RE)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      appendMarkdownSegment(segments, text.slice(lastIndex, index));
    }

    const attachPath = match[1];
    const alias = match[2]?.toLowerCase() as AttachmentKind | undefined;
    const aliasPath = match[3];
    const path = (attachPath ?? aliasPath ?? "").trim();
    if (path) {
      segments.push({
        type: "attachment",
        attachment: {
          kind: attachPath ? "attach" : alias ?? "file",
          path,
        },
      });
    }
    lastIndex = index + match[0].length;
  }

  appendMarkdownSegment(segments, text.slice(lastIndex));
  return segments;
}

function appendMarkdownSegment(segments: MessageSegment[], text: string) {
  if (text.length === 0) return;
  segments.push({ type: "markdown", text });
}

function MarkdownBlocks({ content }: { content: string }) {
  const tokens = useMemo(() => lexer(content), [content]);
  const rendered = renderBlocks(tokens);
  if (!rendered) return null;

  return <div className="space-y-2">{rendered}</div>;
}

function renderBlocks(tokens: Token[] | undefined): ReactNode {
  if (!tokens || tokens.length === 0) return null;

  return tokens.map((token, index) => (
    <MarkdownBlock key={index} token={token} />
  ));
}

function MarkdownBlock({ token }: { token: Token }): ReactNode {
  switch (token.type) {
    case "space":
      return null;

    case "heading": {
      const heading = token as Tokens.Heading;
      return (
        <MarkdownHeading depth={heading.depth}>
          {renderInline(heading.tokens)}
        </MarkdownHeading>
      );
    }

    case "paragraph": {
      const paragraph = token as Tokens.Paragraph;
      return (
        <p className="whitespace-pre-wrap leading-5">
          {renderInline(paragraph.tokens)}
        </p>
      );
    }

    case "text": {
      const text = token as Tokens.Text;
      return (
        <p className="whitespace-pre-wrap leading-5">
          {text.tokens && text.tokens.length > 0
            ? renderInline(text.tokens)
            : text.text}
        </p>
      );
    }

    case "code": {
      const code = token as Tokens.Code;
      return (
        <pre className="max-w-full overflow-x-auto rounded-md border border-neutral-200 bg-neutral-100 px-2.5 py-2 text-xs leading-5 text-neutral-800">
          {code.lang && (
            <div className="mb-1 font-sans text-[11px] text-neutral-500">
              {code.lang}
            </div>
          )}
          <code>{code.text}</code>
        </pre>
      );
    }

    case "blockquote": {
      const quote = token as Tokens.Blockquote;
      return (
        <blockquote className="border-l-2 border-neutral-300 pl-3 text-neutral-700">
          <div className="space-y-2">{renderBlocks(quote.tokens)}</div>
        </blockquote>
      );
    }

    case "list": {
      const list = token as Tokens.List;
      const className = "space-y-1 pl-5 leading-5";
      if (list.ordered) {
        return (
          <ol className={`${className} list-decimal`} start={list.start || 1}>
            {list.items.map((item, index) => (
              <MarkdownListItem key={index} item={item} />
            ))}
          </ol>
        );
      }

      return (
        <ul className={`${className} list-disc`}>
          {list.items.map((item, index) => (
            <MarkdownListItem key={index} item={item} />
          ))}
        </ul>
      );
    }

    case "table": {
      const table = token as Tokens.Table;
      return (
        <div className="max-w-full overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr>
                {table.header.map((cell, index) => (
                  <th
                    key={index}
                    className="border border-neutral-200 bg-neutral-100 px-2 py-1 font-medium"
                  >
                    {renderInline(cell.tokens)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="border border-neutral-200 px-2 py-1 align-top"
                    >
                      {renderInline(cell.tokens)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    case "hr":
      return <hr className="border-neutral-200" />;

    default:
      return token.raw ? (
        <p className="whitespace-pre-wrap leading-5">{token.raw}</p>
      ) : null;
  }
}

function MarkdownHeading({
  depth,
  children,
}: {
  depth: number;
  children: ReactNode;
}) {
  const className =
    depth <= 2
      ? "text-base font-semibold leading-6"
      : "text-sm font-semibold leading-5";

  switch (depth) {
    case 1:
      return <h1 className={className}>{children}</h1>;
    case 2:
      return <h2 className={className}>{children}</h2>;
    case 3:
      return <h3 className={className}>{children}</h3>;
    case 4:
      return <h4 className={className}>{children}</h4>;
    case 5:
      return <h5 className={className}>{children}</h5>;
    default:
      return <h6 className={className}>{children}</h6>;
  }
}

function MarkdownListItem({ item }: { item: Tokens.ListItem }) {
  return (
    <li className="pl-1">
      <div className="space-y-1">
        {item.task && (
          <input
            checked={item.checked}
            className="mr-1 align-middle"
            disabled
            readOnly
            type="checkbox"
          />
        )}
        {item.tokens.length > 0 ? renderBlocks(item.tokens) : item.text}
      </div>
    </li>
  );
}

function renderInline(tokens: Token[] | undefined): ReactNode {
  if (!tokens || tokens.length === 0) return null;

  return tokens.map((token, index) => {
    switch (token.type) {
      case "text": {
        const text = token as Tokens.Text;
        if (text.tokens && text.tokens.length > 0) {
          return <Fragment key={index}>{renderInline(text.tokens)}</Fragment>;
        }
        return <Fragment key={index}>{text.text}</Fragment>;
      }

      case "strong": {
        const strong = token as Tokens.Strong;
        return <strong key={index}>{renderInline(strong.tokens)}</strong>;
      }

      case "em": {
        const em = token as Tokens.Em;
        return <em key={index}>{renderInline(em.tokens)}</em>;
      }

      case "del": {
        const del = token as Tokens.Del;
        return <del key={index}>{renderInline(del.tokens)}</del>;
      }

      case "codespan": {
        const code = token as Tokens.Codespan;
        return (
          <code
            key={index}
            className="rounded bg-neutral-100 px-1 py-0.5 text-[0.9em] text-neutral-800"
          >
            {code.text}
          </code>
        );
      }

      case "link": {
        const link = token as Tokens.Link;
        const href = safeHref(link.href);
        const content = inlineTokenContent(link);
        if (!href) return <Fragment key={index}>{content}</Fragment>;

        return (
          <a
            key={index}
            className="text-blue-700 underline decoration-blue-300 underline-offset-2 hover:text-blue-900"
            href={href}
            rel="noreferrer"
            target={isExternalHref(href) ? "_blank" : undefined}
          >
            {content}
          </a>
        );
      }

      case "image": {
        const image = token as Tokens.Image;
        const src = safeHref(image.href);
        if (!src) return <Fragment key={index}>{image.text || image.href}</Fragment>;

        return (
          <img
            key={index}
            alt={image.text || ""}
            className="my-1 max-h-72 max-w-full rounded-md border border-neutral-200 object-contain"
            src={src}
          />
        );
      }

      case "br":
        return <br key={index} />;

      case "escape": {
        const escape = token as Tokens.Escape;
        return <Fragment key={index}>{escape.text}</Fragment>;
      }

      default:
        return (
          <Fragment key={index}>
            {"text" in token && typeof token.text === "string"
              ? token.text
              : token.raw}
          </Fragment>
        );
    }
  });
}

function inlineTokenContent(token: Tokens.Link): ReactNode {
  const richToken = token as Tokens.Link & { tokens?: Token[] };
  if (richToken.tokens && richToken.tokens.length > 0) {
    return renderInline(richToken.tokens);
  }
  return token.text || token.href;
}

function AttachmentPreview({ attachment }: { attachment: AttachmentMarker }) {
  const url = useMemo(() => attachmentUrl(attachment.path), [attachment.path]);
  const [availability, setAvailability] = useState<
    "checking" | "available" | "unavailable"
  >("checking");
  const displayKind = attachmentDisplayKind(attachment);
  const label = attachmentLabel(attachment.path);

  useEffect(() => {
    let cancelled = false;
    setAvailability("checking");

    fetch(url, { method: "HEAD" })
      .then((response) => {
        if (!cancelled) {
          setAvailability(response.ok ? "available" : "unavailable");
        }
      })
      .catch(() => {
        if (!cancelled) setAvailability("unavailable");
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  if (availability === "unavailable") {
    return <UnavailableAttachment label={label} path={attachment.path} />;
  }

  if (availability === "checking") {
    return <AttachmentShell kind={displayKind} label={label} muted />;
  }

  if (displayKind === "image") {
    return (
      <figure className="max-w-full">
        <img
          alt={label}
          className="max-h-80 max-w-full rounded-md border border-neutral-200 bg-white object-contain"
          onError={() => setAvailability("unavailable")}
          src={url}
        />
        <figcaption className="mt-1 truncate text-xs text-neutral-500">
          {label}
        </figcaption>
      </figure>
    );
  }

  if (displayKind === "audio") {
    return (
      <div className="rounded-md border border-neutral-200 bg-white p-2">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-neutral-600">
          <FileAudio className="size-3.5 shrink-0" />
          <span className="truncate">{label}</span>
        </div>
        <audio
          className="w-full"
          controls
          onError={() => setAvailability("unavailable")}
          preload="metadata"
          src={url}
        />
      </div>
    );
  }

  if (displayKind === "video") {
    return (
      <figure className="max-w-full rounded-md border border-neutral-200 bg-white p-2">
        <video
          className="max-h-80 w-full rounded bg-black"
          controls
          onError={() => setAvailability("unavailable")}
          preload="metadata"
          src={url}
        />
        <figcaption className="mt-1 truncate text-xs text-neutral-500">
          {label}
        </figcaption>
      </figure>
    );
  }

  return (
    <a
      className="inline-flex max-w-full items-center gap-2 rounded-md border border-neutral-200 bg-white px-2.5 py-1.5 text-xs text-neutral-700 hover:border-neutral-300 hover:text-neutral-900"
      download={label}
      href={url}
    >
      <Download className="size-3.5 shrink-0" />
      <span className="truncate">{label}</span>
    </a>
  );
}

function AttachmentShell({
  kind,
  label,
  muted,
}: {
  kind: AttachmentDisplayKind;
  label: string;
  muted?: boolean;
}) {
  const Icon =
    kind === "image"
      ? FileImage
      : kind === "audio"
        ? FileAudio
        : kind === "video"
          ? FileVideo
          : Paperclip;

  return (
    <div
      className={`inline-flex max-w-full items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs ${
        muted
          ? "border-neutral-200 bg-neutral-100 text-neutral-500"
          : "border-neutral-200 bg-white text-neutral-700"
      }`}
    >
      <Icon className="size-3.5 shrink-0" />
      <span className="truncate">{label}</span>
    </div>
  );
}

function UnavailableAttachment({
  label,
  path,
}: {
  label: string;
  path: string;
}) {
  return (
    <div
      className="inline-flex max-w-full items-center gap-2 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs text-amber-800"
      title={path}
    >
      <Paperclip className="size-3.5 shrink-0" />
      <span className="truncate">{label}</span>
      <span className="shrink-0 text-amber-700">unavailable</span>
    </div>
  );
}

function attachmentDisplayKind(
  attachment: AttachmentMarker,
): AttachmentDisplayKind {
  if (attachment.kind === "voice" || attachment.kind === "audio") return "audio";
  if (attachment.kind === "video") return "video";
  if (attachment.kind === "image" || attachment.kind === "photo") return "image";

  const suffix = pathSuffix(attachment.path);
  if (IMAGE_SUFFIXES.has(suffix)) return "image";
  if (AUDIO_SUFFIXES.has(suffix)) return "audio";
  if (VIDEO_SUFFIXES.has(suffix)) return "video";
  return "file";
}

function attachmentLabel(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) || path;
}

function pathSuffix(path: string): string {
  const label = attachmentLabel(path);
  const dot = label.lastIndexOf(".");
  return dot >= 0 ? label.slice(dot).toLowerCase() : "";
}

function attachmentUrl(path: string): string {
  return `/api/attachments?path=${encodeURIComponent(path)}`;
}

function safeHref(value: string): string | undefined {
  const href = value.trim();
  if (!href) return undefined;

  const lower = href.toLowerCase();
  if (
    lower.startsWith("http://") ||
    lower.startsWith("https://") ||
    lower.startsWith("mailto:") ||
    lower.startsWith("tel:")
  ) {
    return href;
  }
  if (
    href.startsWith("#") ||
    (href.startsWith("/") && !href.startsWith("//")) ||
    href.startsWith("./") ||
    href.startsWith("../")
  ) {
    return href;
  }

  return undefined;
}

function isExternalHref(href: string): boolean {
  const lower = href.toLowerCase();
  return lower.startsWith("http://") || lower.startsWith("https://");
}
