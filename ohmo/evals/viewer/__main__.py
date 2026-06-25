"""Run the ohmo trace viewer."""

from __future__ import annotations

import os

from ohmo.evals.viewer.app import create_app


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("OHMO_VIEWER_PORT", "8765"))
    workspace = os.environ.get("OHMO_VIEWER_WORKSPACE") or None
    print(f"Trace viewer running at http://{host}:{port}", flush=True)

    import uvicorn

    uvicorn.run(create_app(workspace), host=host, port=port)


if __name__ == "__main__":
    main()
