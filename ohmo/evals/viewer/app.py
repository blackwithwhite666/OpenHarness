"""Starlette app for browsing ohmo eval traces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ohmo.evals import get_eval_store
from ohmo.evals.viewer.adapter import episode_to_trace_viewer_data, list_prod_traces
from ohmo.evals.viewer.eval_adapter import (
    eval_case_to_trace_viewer_data,
    list_eval_runs,
    list_eval_traces,
)


def create_app(workspace: str | Path | None = None) -> Starlette:
    """Create the read-only trace viewer app."""
    store = get_eval_store(workspace)

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
        run = request.query_params.get("run")
        if run is None or not run.strip():
            return JSONResponse({"detail": "run query parameter is required"}, status_code=400)
        case_id = request.path_params["case_id"]
        try:
            data = eval_case_to_trace_viewer_data(store, run.strip(), case_id)
        except KeyError:
            return JSONResponse({"detail": "eval trace not found"}, status_code=404)
        return JSONResponse(data)

    routes: list[Any] = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/api/traces", traces, methods=["GET"]),
        Route("/api/traces/{trace_id}", trace, methods=["GET"]),
        Route("/api/runs", runs, methods=["GET"]),
        Route("/api/eval-traces", eval_traces, methods=["GET"]),
        Route("/api/eval-traces/{case_id}", eval_trace, methods=["GET"]),
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
