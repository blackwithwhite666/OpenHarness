"""Out-of-process curated-memory service over local AF_UNIX RPC."""

from typing import TYPE_CHECKING, Any

from ohmo.memory_service.client import MemoryServiceClient, MemoryServiceError

if TYPE_CHECKING:
    from ohmo.memory_service.server import MemoryServiceServer, run_server


def __getattr__(name: str) -> Any:
    """Load server exports lazily so ``python -m ...server`` stays warning-free."""
    if name in {"MemoryServiceServer", "run_server"}:
        from ohmo.memory_service.server import MemoryServiceServer, run_server

        value = {
            "MemoryServiceServer": MemoryServiceServer,
            "run_server": run_server,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "MemoryServiceClient",
    "MemoryServiceError",
    "MemoryServiceServer",
    "run_server",
]
