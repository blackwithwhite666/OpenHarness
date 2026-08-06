"""Process-local marker for coordinator-created synthetic turns."""

# This object is intentionally not serializable or user-configurable.  The
# coordinator places it on an in-process bus message; ordinary channel and
# model data can reproduce the spelling of a key but not object identity.
COORDINATOR_TRUST_TOKEN = object()

__all__ = ["COORDINATOR_TRUST_TOKEN"]
