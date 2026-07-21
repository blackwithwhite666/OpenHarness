"""Usage tracking models."""

from __future__ import annotations

from pydantic import BaseModel


class UsageSnapshot(BaseModel):
    """Token usage returned by the model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    """Input tokens served from the provider prompt cache (a cache read/hit)."""
    cache_write_input_tokens: int = 0
    """Input tokens written to the cache (Anthropic cache creation; 0 if unreported)."""

    @property
    def total_tokens(self) -> int:
        """Return the total number of accounted tokens."""
        return self.input_tokens + self.output_tokens
