from unittest.mock import MagicMock, patch
from academic_paper.vector_store import QdrantStore, make_qdrant_id


def test_make_qdrant_id_is_deterministic():
    """同じ引数で同じIDが生成されることを確認"""
    id1 = make_qdrant_id("abc123", 0)
    id2 = make_qdrant_id("abc123", 0)
    assert id1 == id2, "make_qdrant_id should produce deterministic UUIDs"


def test_ensure_collection_creates_when_missing():
    """コレクションが存在しない場合にcreate_collectionが呼ばれることを確認"""
    with patch('academic_paper.vector_store.QdrantClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Simulate no collections exist
        mock_client.get_collections.return_value.collections = []
        
        store = QdrantStore(url="http://test", collection="test-collection")
        store.ensure_collection()
        
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs['collection_name'] == 'test-collection'
        assert call_kwargs['vectors_config'].size == 768


def test_ensure_collection_skips_when_exists():
    """コレクションが既に存在する場合はcreate_collectionが呼ばれないことを確認"""
    with patch('academic_paper.vector_store.QdrantClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Simulate collection already exists
        mock_collection = MagicMock()
        mock_collection.name = "test-collection"
        mock_client.get_collections.return_value.collections = [mock_collection]
        
        store = QdrantStore(url="http://test", collection="test-collection")
        store.ensure_collection()
        
        mock_client.create_collection.assert_not_called()


def test_search_with_paper_id_filter():
    """paper_id_filterを指定してsearchが呼ばれることを確認"""
    with patch('academic_paper.vector_store.QdrantClient') as MockClient:
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
        assert call_kwargs['query_filter'] is not None
        assert len(results) == 1
        assert results[0]['score'] == 0.95
