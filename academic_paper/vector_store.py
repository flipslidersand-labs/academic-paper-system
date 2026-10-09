import asyncio
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams

from academic_paper.config import get_settings
from academic_paper.retry import raise_if_cancelled, to_thread_cancellable, with_retry

PAPER_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class _RetryableQdrantError(Exception):
    """Raised for 5xx/unknown-status Qdrant responses so retries target transient errors only.

    A 4xx UnexpectedResponse (bad filter, missing collection, vector size mismatch, etc.) is
    re-raised as-is and never retried (#306).
    """


def _reraise_qdrant_response(exc: UnexpectedResponse) -> None:
    """Translate UnexpectedResponse into a retryable error, or re-raise 4xx as non-retryable."""
    if exc.status_code is not None and 400 <= exc.status_code < 500:
        raise exc
    raise _RetryableQdrantError(str(exc)) from exc


_QDRANT_RETRYABLE = (_RetryableQdrantError, httpx.NetworkError, httpx.TimeoutException)


def _qdrant_retry(fn):
    """Run fn with retries; after the final attempt re-raise the original UnexpectedResponse.

    _RetryableQdrantError is an internal retry marker. Callers (summarizer fallback,
    server._http_exc_for) classify UnexpectedResponse, so it must not leak out (#472).
    """
    try:
        return with_retry(fn, attempts=3, base_delay=1.0, exceptions=_QDRANT_RETRYABLE)
    except _RetryableQdrantError as exc:
        raise exc.__cause__ from None


_UPSERT_BATCH_MAX = 200  # keep single requests well under qdrant_timeout (#236)


def make_qdrant_id(file_hash: str, chunk_index: int) -> str:
    """UUID5でQdrant point IDを生成（冪等性確保）"""
    return str(uuid.uuid5(PAPER_NS, f"{file_hash}:{chunk_index}"))


class QdrantStore:
    """Qdrant ストア。公開 API は a* 非同期版（aensure_collection/aupsert/adelete_by_paper_id/asearch）と close/aclose。

    _ensure_collection/_upsert/_delete_by_paper_id/_search は to_thread 経由専用の内部実装
    （with_retry の time.sleep を含むため event loop 上で直接呼ぶと停止する #149）。
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        vector_size: int | None = None,
    ):
        s = get_settings()
        self.url = url or s.qdrant_url
        self.api_key = api_key or s.qdrant_api_key or None
        self.collection = collection or s.qdrant_collection
        self.vector_size = vector_size or s.embedding_dim
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=s.qdrant_timeout)

    def _ensure_collection(self) -> None:
        """コレクションが存在しなければ作成（冪等、失敗時3回リトライ #266）
        size=settings.embedding_dim, distance=Cosine。既存コレクションの次元が不一致なら ValueError。
        """

        def _check_size():
            existing = self.client.get_collection(self.collection).config.params.vectors
            existing_size = getattr(existing, "size", None)
            if existing_size is not None and existing_size != self.vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' has vector size {existing_size}, "
                    f"but embedding_dim is {self.vector_size}; recreate the collection or fix EMBEDDING_DIM"
                )

        def _do():
            try:
                collections = self.client.get_collections().collections
                names = [c.name for c in collections]
                if self.collection not in names:
                    try:
                        self.client.create_collection(
                            collection_name=self.collection,
                            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                        )
                    except UnexpectedResponse as exc:
                        # 並行 ingest が先に作成した場合の 409 は成功扱い。次元だけ検証する (#647)
                        if exc.status_code != 409:
                            raise
                        _check_size()
                else:
                    _check_size()
            except UnexpectedResponse as exc:
                _reraise_qdrant_response(exc)

        _qdrant_retry(_do)

    def _upsert(self, points: list[dict]) -> None:
        """チャンクをQdrantにupsertする（失敗時3回リトライ、200件ずつバッチ分割 #236）
        points要素: {"id": str(UUID), "vector": List[float], "payload": dict}
        payload例: {"paper_id": int, "chunk_index": int, "text": str, "file_name": str}

        大容量PDFで数千chunkを一括送信するとqdrant_timeoutを超過しやすく、
        with_retryは同一の巨大リクエストをそのまま再送するだけでサイズ起因の
        タイムアウトは解消しない（embedder.pyの/embed/batch分割と同様の対処）。
        """
        for i in range(0, len(points), _UPSERT_BATCH_MAX):
            raise_if_cancelled()  # skip remaining batches once the caller gave up (#313)
            batch = points[i : i + _UPSERT_BATCH_MAX]
            structs = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in batch]

            def _do(structs=structs):
                try:
                    self.client.upsert(collection_name=self.collection, points=structs)
                except UnexpectedResponse as exc:
                    _reraise_qdrant_response(exc)

            _qdrant_retry(_do)

    def _delete_by_paper_id(self, paper_id: int) -> None:
        """Qdrant から paper_id に属する全ポイントを削除（補償用、#145）。"""
        flt = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])

        def _do():
            try:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=FilterSelector(filter=flt),
                )
            except UnexpectedResponse as exc:
                _reraise_qdrant_response(exc)

        _qdrant_retry(_do)

    def _search(self, query_vector: list[float], limit: int = 10, paper_id_filter: int | None = None) -> list[dict]:
        """ベクトル類似検索（失敗時3回リトライ）
        paper_id_filterが指定された場合はpaper_idでフィルタリング
        Returns: [{"id": str, "score": float, "payload": dict}]
        """
        query_filter = None
        if paper_id_filter is not None:
            query_filter = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id_filter))])

        def _do():
            try:
                return self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                )
            except UnexpectedResponse as exc:
                _reraise_qdrant_response(exc)

        results = _qdrant_retry(_do)
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results.points]

    # ------------------------------------------------------------------
    # Async wrappers — run sync methods in a thread pool so event-loop
    # callers don't block (#149). with_retry uses time.sleep which is
    # safe inside a thread but would stall the loop if called directly.
    # A cancelled caller (e.g. wait_for timeout) cannot stop a running thread;
    # to_thread_cancellable only stops further retries/batches (#313). The
    # in-flight request still runs until settings.qdrant_timeout.
    # ------------------------------------------------------------------

    async def aupsert(self, points: list[dict]) -> None:
        await to_thread_cancellable(self._upsert, points)

    async def adelete_by_paper_id(self, paper_id: int) -> None:
        await to_thread_cancellable(self._delete_by_paper_id, paper_id)

    async def asearch(
        self, query_vector: list[float], limit: int = 10, paper_id_filter: int | None = None
    ) -> list[dict]:
        return await to_thread_cancellable(self._search, query_vector, limit, paper_id_filter)

    async def aensure_collection(self) -> None:
        await to_thread_cancellable(self._ensure_collection)

    def ping(self) -> None:
        """Qdrant への疎通確認。到達不能なら例外を送出する（/health・起動プローブ用）。"""
        self.client.get_collections()

    async def aping(self) -> None:
        await asyncio.to_thread(self.ping)

    def count_points(self) -> int | None:
        """このストアのコレクションのポイント数を返す（/stats 用）。"""
        return self.client.get_collection(self.collection).points_count

    def close(self) -> None:
        """QdrantClient のコネクションプールを解放する（#228）。"""
        self.client.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
