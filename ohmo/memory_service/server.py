"""AF_UNIX RPC server that owns the curated-memory catalog."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import stat
from pathlib import Path
from typing import Mapping

from ohmo.memory_backend import CatalogMemoryBackend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.protocol import (
    MemoryServiceProtocolError,
    load_secret_file,
    memory_entry_to_dict,
    memory_hit_to_dict,
    memory_op_result_to_dict,
    read_frame,
    verify_capability,
    write_frame,
)

_AUTHORIZED_OPERATIONS = frozenset(
    {"list", "get", "search", "add", "update", "remove", "render_prompt", "append_turn"}
)


class MemoryServiceServer:
    """Lifecycle wrapper for one local-only memory service."""

    def __init__(
        self,
        socket_path: str | Path,
        workspace: str | Path,
        secret_file: str | Path,
    ) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.workspace = Path(workspace).expanduser()
        self.secret_file = Path(secret_file).expanduser()
        self._secret: bytes | None = None
        self._backend: CatalogMemoryBackend | None = None
        self._server: asyncio.AbstractServer | None = None

    @property
    def backend(self) -> CatalogMemoryBackend:
        """Return the owned backend after the server has started."""
        if self._backend is None:
            raise RuntimeError("memory service has not started")
        return self._backend

    async def start(self) -> "MemoryServiceServer":
        """Load authority state and bind the owner-only Unix socket."""
        if self._server is not None:
            raise RuntimeError("memory service is already running")
        secret = load_secret_file(self.secret_file)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._remove_stale_socket()
        backend = CatalogMemoryBackend(MemoryCatalog(self.workspace), self.workspace)
        try:
            server = await asyncio.start_unix_server(
                self._handle_connection,
                path=str(self.socket_path),
            )
            os.chmod(self.socket_path, 0o600)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                self.socket_path.unlink()
            raise
        self._secret = secret
        self._backend = backend
        self._server = server
        return self

    async def close(self) -> None:
        """Stop accepting requests and remove the service socket."""
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        self._secret = None
        self._backend = None
        with contextlib.suppress(FileNotFoundError):
            if self.socket_path.is_socket():
                self.socket_path.unlink()

    async def serve_forever(self) -> None:
        """Serve until cancelled, starting first when necessary."""
        if self._server is None:
            await self.start()
        server = self._server
        if server is None:  # pragma: no cover - guarded by start()
            raise RuntimeError("memory service failed to start")
        try:
            async with server:
                await server.serve_forever()
        finally:
            await self.close()

    async def __aenter__(self) -> "MemoryServiceServer":
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _remove_stale_socket(self) -> None:
        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(metadata.st_mode):
            raise RuntimeError(f"refusing to replace non-socket path: {self.socket_path}")
        self.socket_path.unlink()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                try:
                    request = await read_frame(reader)
                except asyncio.IncompleteReadError:
                    break
                except MemoryServiceProtocolError:
                    await write_frame(writer, {"ok": False, "error": "invalid request"})
                    break
                response = await self._handle_request(request)
                await write_frame(writer, response)
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await writer.wait_closed()

    async def _handle_request(self, request: Mapping[str, object]) -> dict[str, object]:
        token = request.get("token")
        op = request.get("op")
        secret = self._secret
        if (
            secret is None
            or not isinstance(token, str)
            or not isinstance(op, str)
            or op not in _AUTHORIZED_OPERATIONS
            or not verify_capability(secret, token, op)
        ):
            return {"ok": False, "error": "unauthorized"}

        args = request.get("args")
        if not isinstance(args, dict):
            return {"ok": False, "error": "invalid arguments"}
        try:
            result = await self._dispatch(op, args)
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "invalid arguments"}
        except Exception:
            return {"ok": False, "error": "internal error"}
        return {"ok": True, "result": result}

    async def _dispatch(self, op: str, args: Mapping[str, object]) -> object:
        backend = self.backend
        if op == "list":
            return [memory_entry_to_dict(entry) for entry in await backend.list()]
        if op == "get":
            entry = await backend.get(_str_arg(args, "name"))
            return None if entry is None else memory_entry_to_dict(entry)
        if op == "search":
            hits = await backend.search(_str_arg(args, "query"), _int_arg(args, "top_k"))
            return [memory_hit_to_dict(hit) for hit in hits]
        if op == "add":
            result = await backend.add(_str_arg(args, "title"), _str_arg(args, "content"))
            return memory_op_result_to_dict(result)
        if op == "update":
            result = await backend.update(_str_arg(args, "name"), _str_arg(args, "content"))
            return memory_op_result_to_dict(result)
        if op == "remove":
            return memory_op_result_to_dict(await backend.remove(_str_arg(args, "name")))
        if op == "render_prompt":
            budget = args.get("budget")
            if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool)):
                raise TypeError("budget must be an integer or null")
            return await backend.render_prompt(budget)
        if op == "append_turn":
            await backend.append_turn(_str_arg(args, "role"), _str_arg(args, "text"))
            return None
        raise ValueError(f"unsupported operation {op!r}")


def _str_arg(args: Mapping[str, object], name: str) -> str:
    value = args[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _int_arg(args: Mapping[str, object], name: str) -> int:
    value = args[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone service's command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path, help="AF_UNIX socket path")
    parser.add_argument("--workspace", required=True, type=Path, help="ohmo workspace")
    parser.add_argument("--secret-file", required=True, type=Path, help="0600 capability secret")
    return parser


async def run_server(
    socket_path: str | Path,
    workspace: str | Path,
    secret_file: str | Path,
) -> None:
    """Run a memory service until cancellation."""
    service = MemoryServiceServer(socket_path, workspace, secret_file)
    await service.serve_forever()


def main() -> None:
    """Run the memory service command-line entry point."""
    args = build_argument_parser().parse_args()
    asyncio.run(run_server(args.socket, args.workspace, args.secret_file))


if __name__ == "__main__":
    main()


__all__ = ["MemoryServiceServer", "main", "run_server"]
