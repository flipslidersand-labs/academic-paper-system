"""Shared HTTP status classification (#739).

Two predicates, deliberately NOT complements for 429:

- is_client_error: any 4xx (incl. 429). Used where a 4xx means a standing
  misconfiguration that must propagate (summarizer fallback #305, Qdrant no-retry #306).
- is_retryable_status: 5xx and 429. Used by the embedder to decide what to retry.

Current behaviour is preserved: Qdrant/summarizer treat 429 as a client error
(not retried / not silently degraded), while the embedder retries 429.
"""


def is_client_error(status: int | None) -> bool:
    """True for 4xx; False for None (unknown status), 1xx-3xx and 5xx."""
    return status is not None and 400 <= status < 500


def is_retryable_status(status: int | None) -> bool:
    """True for 5xx and 429 (transient); False for None and other statuses."""
    return status is not None and (status >= 500 or status == 429)
