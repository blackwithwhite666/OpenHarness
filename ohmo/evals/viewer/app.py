"""Starlette app for browsing ohmo eval traces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ohmo.evals import get_eval_store
from ohmo.evals.viewer.adapter import (
    episode_session_conversation,
    episode_to_trace_viewer_data,
    list_prod_traces,
)
from ohmo.evals.viewer.eval_adapter import (
    eval_case_conversation,
    eval_case_to_trace_viewer_data,
    list_eval_runs,
    list_eval_traces,
)
from ohmo.workspace import get_attachments_dir

_INLINE_MEDIA_TYPES = {
    ".apng": "image/apng",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".ogv": "video/ogg",
    ".webm": "video/webm",
}
_DOWNLOAD_ATTACHMENT_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".htm",
    ".html",
    ".json",
    ".log",
    ".md",
    ".pdf",
    ".ppt",
    ".pptx",
    ".tar",
    ".tsv",
    ".txt",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


def create_app(workspace: str | Path | None = None) -> Starlette:
    """Create the read-only trace viewer app."""
    store = get_eval_store(workspace)
    attachment_roots = _attachment_roots(workspace)

    async def healthz(_: Any) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def traces(request: Any) -> JSONResponse:
        params = request.query_params
        source = params.get("source", "prod")
        if source != "prod":
            return JSONResponse({"detail": f"unsupported trace source: {source}"}, status_code=400)
        return JSONResponse(
            list_prod_traces(
                store,
                q=params.get("q") or None,
                limit=_int_param(params.get("limit"), default=50),
                offset=_int_param(params.get("offset"), default=0),
            )
        )

    async def trace(request: Any) -> JSONResponse:
        trace_id = request.path_params["trace_id"]
        try:
            data = episode_to_trace_viewer_data(store, trace_id)
        except KeyError:
            return JSONResponse({"detail": "trace not found"}, status_code=404)
        return JSONResponse(data)

    async def runs(_: Any) -> JSONResponse:
        return JSONResponse(list_eval_runs(store))

    async def eval_traces(request: Any) -> JSONResponse:
        run = request.query_params.get("run")
        if run is None or not run.strip():
            return JSONResponse({"detail": "run query parameter is required"}, status_code=400)
        try:
            data = list_eval_traces(store, run.strip())
        except KeyError:
            return JSONResponse({"detail": "eval run not found"}, status_code=404)
        return JSONResponse(data)

    async def eval_trace(request: Any) -> JSONResponse:
        params = request.query_params
        run = params.get("run")
        if run is None or not run.strip():
            return JSONResponse({"detail": "run query parameter is required"}, status_code=400)
        case_id = request.path_params["case_id"]
        sample = max(0, _int_param(params.get("sample"), default=0))
        try:
            data = eval_case_to_trace_viewer_data(store, run.strip(), case_id, sample=sample)
        except KeyError:
            return JSONResponse({"detail": "eval trace not found"}, status_code=404)
        return JSONResponse(data)

    async def session(request: Any) -> JSONResponse:
        episode_id = request.path_params["episode_id"]
        try:
            data = episode_session_conversation(store, episode_id)
        except KeyError:
            return JSONResponse({"detail": "episode not found"}, status_code=404)
        return JSONResponse(data)

    async def eval_conversation(request: Any) -> JSONResponse:
        params = request.query_params
        run = params.get("run")
        if run is None or not run.strip():
            return JSONResponse({"detail": "run query parameter is required"}, status_code=400)
        case_id = request.path_params["case_id"]
        sample = max(0, _int_param(params.get("sample"), default=0))
        try:
            data = eval_case_conversation(store, run.strip(), case_id, sample=sample)
        except KeyError:
            return JSONResponse({"detail": "eval case not found"}, status_code=404)
        return JSONResponse(data)

    async def attachment(request: Any) -> FileResponse | JSONResponse:
        raw_path = request.query_params.get("path")
        if raw_path is None or not raw_path.strip() or "\x00" in raw_path:
            return JSONResponse({"detail": "attachment path is required"}, status_code=400)

        try:
            path = Path(raw_path).expanduser()
        except RuntimeError:
            return JSONResponse({"detail": "invalid attachment path"}, status_code=400)
        if not path.is_absolute():
            return JSONResponse({"detail": "attachment path must be absolute"}, status_code=403)

        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return JSONResponse({"detail": "attachment not found"}, status_code=404)

        if not resolved.is_file():
            return JSONResponse({"detail": "attachment not found"}, status_code=404)
        if not _is_allowed_attachment_path(resolved, attachment_roots):
            return JSONResponse({"detail": "attachment path is not allowed"}, status_code=403)

        suffix = resolved.suffix.lower()
        media_type = _INLINE_MEDIA_TYPES.get(suffix)
        content_disposition_type = "inline"
        if media_type is None:
            if suffix not in _DOWNLOAD_ATTACHMENT_SUFFIXES:
                return JSONResponse({"detail": "attachment type is not allowed"}, status_code=403)
            media_type = "application/octet-stream"
            content_disposition_type = "attachment"

        response = FileResponse(
            resolved,
            filename=resolved.name,
            media_type=media_type,
            content_disposition_type=content_disposition_type,
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    routes: list[Any] = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/api/traces", traces, methods=["GET"]),
        Route("/api/traces/{trace_id}", trace, methods=["GET"]),
        Route("/api/runs", runs, methods=["GET"]),
        Route("/api/eval-traces", eval_traces, methods=["GET"]),
        Route("/api/eval-traces/{case_id}", eval_trace, methods=["GET"]),
        Route("/api/session/{episode_id}", session, methods=["GET"]),
        Route("/api/eval-conversation/{case_id}", eval_conversation, methods=["GET"]),
        Route("/api/attachments", attachment, methods=["GET", "HEAD"]),
    ]

    # Static SPA: env override (used on the server, where the repo tree is absent
    # because the package is pip-installed) falling back to the local repo build.
    static_override = os.getenv("OHMO_VIEWER_STATIC_DIR")
    static_dir = (
        Path(static_override).expanduser()
        if static_override
        else _repo_root() / "frontend" / "trace-viewer" / "dist"
    )
    if static_dir.exists():
        routes.append(Mount("/", StaticFiles(directory=static_dir, html=True), name="trace-viewer"))

    return Starlette(routes=routes)


def _int_param(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _attachment_roots(workspace: str | Path | None) -> tuple[Path, ...]:
    return (
        Path("/tmp").resolve(strict=False),
        get_attachments_dir(workspace).resolve(strict=False),
    )


def _is_allowed_attachment_path(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)
