import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from academic_paper.config import settings

PAPER_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

def make_qdrant_id(file_hash: str, chunk_index: int) -> str:
    """UUID5でQdrant point IDを生成（冪等性確保）"""
    return str(uuid.uuid5(PAPER_NS, f"{file_hash}:{chunk_index}"))

class QdrantStore:
    def __init__(self, url: str | None = None, api_key: str | None = None, collection: str | None = None):
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key or None
        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

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
        """チャンクをQdrantにupsertする
        points要素: {"id": str(UUID), "vector": List[float], "payload": dict}
        payload例: {"paper_id": int, "chunk_index": int, "text": str, "file_name": str}
        """
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points],
        )

    def search(self, query_vector: list[float], limit: int = 10, paper_id_filter: int | None = None) -> list[dict]:
        """ベクトル類似検索
        paper_id_filterが指定された場合はpaper_idでフィルタリング
        Returns: [{"id": str, "score": float, "payload": dict}]
        """
        query_filter = None
        if paper_id_filter is not None:
            query_filter = Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id_filter))]
            )
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]
