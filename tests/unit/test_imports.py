from app.retrieval.hybrid_retriever import HybridEnsembleRetriever
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.core.config import settings

def test_imports():
    assert HybridEnsembleRetriever
    assert CrossEncoderReranker

def test_config_loads():
    assert isinstance(settings.log_level, str)
