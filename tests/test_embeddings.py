try:
    import pytest
except ImportError:
    pytest = None
from app.embeddings.embedder import get_embeddings

def test_embeddings_init():
    embedder = get_embeddings()
    assert embedder is not None
