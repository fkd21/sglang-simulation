"""Decode Continuation mechanism.

When enabled, after prefill completes on a prefill instance, the request
can continue generating M tokens on that same prefill instance before
being transferred to a decode instance. This reduces KV transfer overhead
for requests that would only generate a few tokens.
"""

from __future__ import annotations

from simulation.request.request import SimReq


class DecodeContinuation:
    """Implements decode continuation with parameter M.

    M controls how many decode tokens are generated on the prefill instance
    before transferring to decode:
    - M=0: No continuation (baseline) - transfer immediately after prefill
    - M=10: Generate 10 tokens on prefill instance, then transfer
    - M=inf: Generate all tokens on prefill (no P/D disaggregation)
    """

    def __init__(self, M: int = 0):
        """Initialize continuation.

        Args:
            M: Number of decode tokens to generate on prefill instance
        """
        assert M >= 0, f"M must be >= 0, got {M}"
        self.M = M

    @property
    def enabled(self) -> bool:
        return self.M > 0

    def tokens_to_continue(self, req: SimReq) -> int:
        """Compute how many tokens to generate on prefill instance.

        Args:
            req: Request that just completed prefill

        Returns:
            Number of tokens to generate on prefill before transferring
        """
        if not self.enabled:
            return 0

        # Don't generate more than what's needed
        remaining = req.tokens_remaining()
        continuation = min(self.M, remaining)

        return continuation
