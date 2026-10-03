import asyncio
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from academic_paper.vector_store import _QDRANT_RETRYABLE, QdrantStore, make_qdrant_id


def test_make_qdrant_id_is_deterministic():
    """同じ引数で同じIDが生成されることを確認"""
    id1 = make_qdrant_id("abc123", 0)
    id2 = make_qdrant_id("abc123", 0)
    assert id1 == id2, "make_qdrant_id should produce deterministic UUIDs"


def test_ensure_collection_creates_when_missing():
    """コレクションが存在しない場合にcreate_collectionが呼ばれることを確認"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Simulate no collections exist
        mock_client.get_collections.return_value.collections = []

        store = QdrantStore(url="http://test", collection="test-collection")
        store.ensure_collection()

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["vectors_config"].size == 768


def test_ensure_collection_skips_when_exists():
    """コレクションが既に存在する場合はcreate_collectionが呼ばれないことを確認"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Simulate collection already exists
        mock_collection = MagicMock()
        mock_collection.name = "test-collection"
        mock_client.get_collections.return_value.collections = [mock_collection]
        mock_client.get_collection.return_value.config.params.vectors.size = 768

        store = QdrantStore(url="http://test", collection="test-collection")
        store.ensure_collection()

        mock_client.create_collection.assert_not_called()


def test_ensure_collection_uses_configured_vector_size():
    """vector_size 指定がコレクション作成に反映される (#494)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        QdrantStore(url="http://test", collection="c", vector_size=1024).ensure_collection()

        assert mock_client.create_collection.call_args[1]["vectors_config"].size == 1024


def test_ensure_collection_raises_on_dimension_mismatch():
    """既存コレクションの次元が不一致なら ValueError (#494)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "c"
        mock_client.get_collections.return_value.collections = [mock_collection]
        mock_client.get_collection.return_value.config.params.vectors.size = 384

        store = QdrantStore(url="http://test", collection="c", vector_size=768)
        with pytest.raises(ValueError, match="vector size 384"):
            store.ensure_collection()


def test_ensure_collection_passes_retry_params():
    """ensure_collection が with_retry に attempts=3 と retryable exceptions を渡すことを確認 (#266)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        with patch("academic_paper.vector_store.with_retry") as mock_retry:
            mock_retry.return_value = None

            store = QdrantStore(url="http://test", collection="test-collection")
            store.ensure_collection()

        mock_retry.assert_called_once()
        _, kw = mock_retry.call_args
        assert kw["attempts"] == 3
        assert kw["exceptions"] == _QDRANT_RETRYABLE


def test_search_with_paper_id_filter():
    """paper_id_filterを指定してsearchが呼ばれることを確認"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        # Simulate query_points results (qdrant-client 1.18+)
        mock_result = MagicMock()
        mock_result.id = "123"
        mock_result.score = 0.95
        mock_result.payload = {"paper_id": 1, "text": "test"}
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_result]
        mock_client.query_points.return_value = mock_query_response

        store = QdrantStore(url="http://test", collection="test-collection")
        results = store.search([0.1] * 768, limit=10, paper_id_filter=1)

        # Verify query_points was called with filter
        assert mock_client.query_points.called
        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None
        assert len(results) == 1
        assert results[0]["score"] == 0.95


def test_upsert_calls_qdrant_client():
    """upsert がポイントを Qdrant に送信することを確認 (#201)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        store = QdrantStore(url="http://test", collection="test-collection")
        points = [
            {
                "id": "aaaaaaaa-0000-0000-0000-000000000001",
                "vector": [0.1] * 768,
                "payload": {"paper_id": 1, "chunk_index": 0, "text": "hello"},
            }
        ]
        store.upsert(points)

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert len(call_kwargs["points"]) == 1


def test_upsert_splits_into_batches():
    """points が _UPSERT_BATCH_MAX を超える場合、複数回に分割して upsert されることを確認 (#236)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        store = QdrantStore(url="http://test", collection="test-collection")
        points = [
            {
                "id": f"aaaaaaaa-0000-0000-0000-{i:012d}",
                "vector": [0.1] * 768,
                "payload": {"paper_id": 1, "chunk_index": i, "text": "hello"},
            }
            for i in range(450)
        ]
        store.upsert(points)

        # 200件ずつ: 200, 200, 50 の3回に分割される
        assert mock_client.upsert.call_count == 3
        sizes = [len(call.kwargs["points"]) for call in mock_client.upsert.call_args_list]
        assert sizes == [200, 200, 50]
        for call in mock_client.upsert.call_args_list:
            assert call.kwargs["collection_name"] == "test-collection"


def test_upsert_single_batch_when_under_limit():
    """points が上限未満の場合は1回だけ upsert が呼ばれることを確認 (#236)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        store = QdrantStore(url="http://test", collection="test-collection")
        points = [
            {
                "id": f"aaaaaaaa-0000-0000-0000-{i:012d}",
                "vector": [0.1] * 768,
                "payload": {"paper_id": 1, "chunk_index": i, "text": "hello"},
            }
            for i in range(50)
        ]
        store.upsert(points)

        mock_client.upsert.assert_called_once()
        assert len(mock_client.upsert.call_args.kwargs["points"]) == 50


def test_upsert_passes_retry_params():
    """upsert が with_retry に attempts=3 と retryable exceptions を渡すことを確認 (#201)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        with patch("academic_paper.vector_store.with_retry") as mock_retry:
            mock_retry.return_value = None

            store = QdrantStore(url="http://test", collection="test-collection")
            points = [{"id": "aaa", "vector": [0.1] * 768, "payload": {}}]
            store.upsert(points)

        mock_retry.assert_called_once()
        _, kw = mock_retry.call_args
        assert kw["attempts"] == 3


def test_delete_by_paper_id_calls_qdrant_client():
    """delete_by_paper_id が paper_id フィルタで Qdrant を呼ぶことを確認 (#201)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        store = QdrantStore(url="http://test", collection="test-collection")
        store.delete_by_paper_id(42)

        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        # points_selector は FilterSelector で paper_id=42 フィルタを持つ
        assert call_kwargs["points_selector"] is not None


def test_delete_by_paper_id_retries_on_network_error():
    """delete_by_paper_id が NetworkError 時に with_retry でリトライすることを確認 (#201)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        with patch("academic_paper.vector_store.with_retry") as mock_retry:
            mock_retry.return_value = None

            store = QdrantStore(url="http://test", collection="test-collection")
            store.delete_by_paper_id(7)

        mock_retry.assert_called_once()
        _, kw = mock_retry.call_args
        assert kw["attempts"] == 3


def test_search_does_not_retry_on_4xx():
    """A 400 UnexpectedResponse (e.g. bad filter) is not retried — fails on the first attempt (#306)."""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        call_count = [0]

        def _raise(*args, **kwargs):
            call_count[0] += 1
            raise UnexpectedResponse(status_code=400, reason_phrase="Bad Request", content=b"", headers={})

        mock_client.query_points.side_effect = _raise

        store = QdrantStore(url="http://test", collection="test-collection")
        try:
            store.search([0.1] * 768, limit=10)
        except UnexpectedResponse:
            pass
        else:
            raise AssertionError("expected UnexpectedResponse to propagate")

        assert call_count[0] == 1


def test_search_retries_on_5xx():
    """A 500 UnexpectedResponse is retried up to `attempts` times (#306)."""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        call_count = [0]

        def _raise(*args, **kwargs):
            call_count[0] += 1
            raise UnexpectedResponse(status_code=500, reason_phrase="Server Error", content=b"", headers={})

        mock_client.query_points.side_effect = _raise

        with patch("academic_paper.retry.time.sleep"):
            store = QdrantStore(url="http://test", collection="test-collection")
            try:
                store.search([0.1] * 768, limit=10)
            except UnexpectedResponse as exc:
                # The internal retry marker must not leak; the original error surfaces (#472).
                assert exc.status_code == 500
            else:
                raise AssertionError("expected UnexpectedResponse to propagate after retries")

        assert call_count[0] == 3


def test_close_closes_underlying_qdrant_client():
    """close() が内部の QdrantClient.close() を呼ぶことを確認 (#228)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        store = QdrantStore(url="http://test", collection="test-collection")
        store.close()

        mock_client.close.assert_called_once()


def test_aclose_closes_underlying_qdrant_client():
    """aclose() が内部の QdrantClient.close() を呼ぶことを確認 (#228)"""
    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        store = QdrantStore(url="http://test", collection="test-collection")
        asyncio.run(store.aclose())

        mock_client.close.assert_called_once()


def test_aupsert_cancel_skips_remaining_batches():
    """#313: once the awaiting task is cancelled, the worker thread stops after the
    in-flight batch instead of upserting every remaining batch."""
    import threading

    from academic_paper.vector_store import _UPSERT_BATCH_MAX

    in_first = threading.Event()
    release = threading.Event()
    done = threading.Event()
    upserts = []

    def slow_upsert(**kwargs):
        upserts.append(1)
        in_first.set()
        release.wait(5)

    with patch("academic_paper.vector_store.QdrantClient") as MockClient:  # noqa: N806
        mock_client = MagicMock()
        mock_client.upsert.side_effect = slow_upsert
        MockClient.return_value = mock_client
        store = QdrantStore(url="http://test", collection="c")

        points = [{"id": str(i), "vector": [0.0], "payload": {}} for i in range(_UPSERT_BATCH_MAX * 3)]

        async def scenario():
            task = asyncio.create_task(store.aupsert(points))
            await asyncio.to_thread(in_first.wait, 5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            release.set()
            await asyncio.sleep(0.3)
            done.set()

        asyncio.run(scenario())

    assert done.is_set()
    assert len(upserts) == 1
