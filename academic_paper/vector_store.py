import asyncio
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams

from academic_paper.config import settings
from academic_paper.retry import with_retry

PAPER_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

_QDRANT_RETRYABLE = (UnexpectedResponse, httpx.NetworkError, httpx.TimeoutException)
_UPSERT_BATCH_MAX = 200  # keep single requests well under qdrant_timeout (#236)


def make_qdrant_id(file_hash: str, chunk_index: int) -> str:
    """UUID5でQdrant point IDを生成（冪等性確保）"""
    return str(uuid.uuid5(PAPER_NS, f"{file_hash}:{chunk_index}"))


class QdrantStore:
    def __init__(self, url: str | None = None, api_key: str | None = None, collection: str | None = None):
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key or None
        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=settings.qdrant_timeout)

    def ensure_collection(self) -> None:
        """コレクションが存在しなければ作成（冪等）
        size=768, distance=Cosine
        """
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def upsert(self, points: list[dict]) -> None:
        """チャンクをQdrantにupsertする（失敗時3回リトライ、200件ずつバッチ分割 #236）
        points要素: {"id": str(UUID), "vector": List[float], "payload": dict}
        payload例: {"paper_id": int, "chunk_index": int, "text": str, "file_name": str}

        大容量PDFで数千chunkを一括送信するとqdrant_timeoutを超過しやすく、
        with_retryは同一の巨大リクエストをそのまま再送するだけでサイズ起因の
        タイムアウトは解消しない（embedder.pyの/embed/batch分割と同様の対処）。
        """
        for i in range(0, len(points), _UPSERT_BATCH_MAX):
            batch = points[i : i + _UPSERT_BATCH_MAX]
            structs = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in batch]

            def _do(structs=structs):
                self.client.upsert(collection_name=self.collection, points=structs)

            with_retry(_do, attempts=3, base_delay=1.0, exceptions=_QDRANT_RETRYABLE)

    def delete_by_paper_id(self, paper_id: int) -> None:
        """Qdrant から paper_id に属する全ポイントを削除（補償用、#145）。"""
        flt = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])

        def _do():
            self.client.delete(
                collection_name=self.collection,
                points_selector=FilterSelector(filter=flt),
            )

        with_retry(_do, attempts=3, base_delay=1.0, exceptions=_QDRANT_RETRYABLE)

    def search(self, query_vector: list[float], limit: int = 10, paper_id_filter: int | None = None) -> list[dict]:
        """ベクトル類似検索（失敗時3回リトライ）
        paper_id_filterが指定された場合はpaper_idでフィルタリング
        Returns: [{"id": str, "score": float, "payload": dict}]
        """
        query_filter = None
        if paper_id_filter is not None:
            query_filter = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id_filter))])

        def _do():
            return self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
            )

        results = with_retry(_do, attempts=3, base_delay=1.0, exceptions=_QDRANT_RETRYABLE)
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results.points]

    # ------------------------------------------------------------------
    # Async wrappers — run sync methods in a thread pool so event-loop
    # callers don't block (#149). with_retry uses time.sleep which is
    # safe inside a thread but would stall the loop if called directly.
    # ------------------------------------------------------------------

    async def aupsert(self, points: list[dict]) -> None:
        await asyncio.to_thread(self.upsert, points)

    async def adelete_by_paper_id(self, paper_id: int) -> None:
        await asyncio.to_thread(self.delete_by_paper_id, paper_id)

    async def asearch(
        self, query_vector: list[float], limit: int = 10, paper_id_filter: int | None = None
    ) -> list[dict]:
        return await asyncio.to_thread(self.search, query_vector, limit, paper_id_filter)

    async def aensure_collection(self) -> None:
        await asyncio.to_thread(self.ensure_collection)

    def close(self) -> None:
        """QdrantClient のコネクションプールを解放する（#228）。"""
        self.client.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
