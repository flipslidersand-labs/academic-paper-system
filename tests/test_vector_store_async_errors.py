"""Async wrapper delegation and 4xx/5xx branch tests for QdrantStore (#353).

Complements tests/test_vector_store.py. No network and no real Qdrant: QdrantClient is patched.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from academic_paper.vector_store import QdrantStore

_POINTS = [{"id": "aaaaaaaa-0000-0000-0000-000000000001", "vector": [0.1] * 768, "payload": {"paper_id": 1}}]


@pytest.fixture
def client():
    with patch("academic_paper.vector_store.QdrantClient") as mock_cls:
        yield mock_cls.return_value


@pytest.fixture
def store(client):
    return QdrantStore(url="http://test", collection="test-collection")


@pytest.fixture(autouse=True)
def _no_retry_sleep():
    with patch("academic_paper.retry.time.sleep"):
        yield


def _unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status_code=status, reason_phrase="err", content=b"", headers={})


# --- async wrappers delegate to the sync methods with the right arguments -------------------


def test_aupsert_delegates_to_upsert(store):
    with patch.object(store, "upsert") as m:
        asyncio.run(store.aupsert(_POINTS))
    m.assert_called_once_with(_POINTS)


def test_adelete_by_paper_id_delegates(store):
    with patch.object(store, "delete_by_paper_id") as m:
        asyncio.run(store.adelete_by_paper_id(42))
    m.assert_called_once_with(42)


def test_asearch_delegates_with_positional_order(store):
    vec = [0.5] * 768
    with patch.object(store, "search", return_value=[{"id": "1"}]) as m:
        result = asyncio.run(store.asearch(vec, 5, 9))
    assert result == [{"id": "1"}]
    m.assert_called_once_with(vec, 5, 9)  # (query_vector, limit, paper_id_filter)


def test_asearch_defaults(store):
    vec = [0.5] * 768
    with patch.object(store, "search", return_value=[]) as m:
        asyncio.run(store.asearch(vec))
    m.assert_called_once_with(vec, 10, None)


def test_aensure_collection_delegates(store):
    with patch.object(store, "ensure_collection") as m:
        asyncio.run(store.aensure_collection())
    m.assert_called_once_with()


def test_async_wrappers_propagate_exceptions(store):
    with patch.object(store, "upsert", side_effect=ValueError("boom")), pytest.raises(ValueError, match="boom"):
        asyncio.run(store.aupsert(_POINTS))


# --- 4xx is not retried, 5xx is retried attempts=3 times ------------------------------------

_OPERATIONS = {
    "ensure_collection": (lambda s: s.ensure_collection(), "get_collections"),
    "upsert": (lambda s: s.upsert(_POINTS), "upsert"),
    "delete_by_paper_id": (lambda s: s.delete_by_paper_id(1), "delete"),
}


@pytest.mark.parametrize("op", list(_OPERATIONS))
def test_4xx_propagates_immediately_without_retry(op, store, client):
    call, client_method = _OPERATIONS[op]
    getattr(client, client_method).side_effect = _unexpected(400)

    with pytest.raises(UnexpectedResponse) as exc_info:
        call(store)

    assert exc_info.value.status_code == 400
    assert getattr(client, client_method).call_count == 1


@pytest.mark.parametrize("op", list(_OPERATIONS))
def test_5xx_is_retried_three_times_then_propagates(op, store, client):
    call, client_method = _OPERATIONS[op]
    getattr(client, client_method).side_effect = _unexpected(500)

    # Exception type after exhausting retries is an implementation detail (_RetryableQdrantError
    # today; the original UnexpectedResponse once #472 lands) — assert on the retry count.
    with pytest.raises(Exception, match="500|err"):
        call(store)

    assert getattr(client, client_method).call_count == 3


@pytest.mark.parametrize("op", list(_OPERATIONS))
def test_5xx_then_success_recovers(op, store, client):
    call, client_method = _OPERATIONS[op]
    m = getattr(client, client_method)
    if op == "ensure_collection":
        ok = MagicMock(collections=[MagicMock()])
        ok.collections[0].name = "test-collection"
    else:
        ok = MagicMock()
    m.side_effect = [_unexpected(503), ok]

    call(store)

    assert m.call_count == 2


@pytest.mark.parametrize("op", list(_OPERATIONS))
def test_status_none_is_treated_as_retryable(op, store, client):
    call, client_method = _OPERATIONS[op]
    exc = UnexpectedResponse(status_code=None, reason_phrase="", content=b"", headers={})
    getattr(client, client_method).side_effect = exc

    with pytest.raises(Exception):  # noqa: B017
        call(store)

    assert getattr(client, client_method).call_count == 3
