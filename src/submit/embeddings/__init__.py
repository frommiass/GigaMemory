"""
Улучшенная система векторного поиска для GigaMemory
Версия 2.0 с расширенной функциональностью
"""

from .improved_vector_search import ImprovedEmbeddingEngine, EmbeddingConfig, PoolingStrategy
from .improved_vector_store import ImprovedVectorStore
from .vector_models import (
    SimilarityMetric, VectorDocument, SearchResult, 
    SearchQuery, BatchSearchQuery, HybridSearchQuery,
    VectorStoreStats, EmbeddingStats, PerformanceMetrics
)
from .vector_utils import benchmark_performance, compare_models, stress_test, optimize_store_config

__version__ = "2.0.0"
__author__ = "GigaMemory Team"

__all__ = [
    # Основные классы
    "ImprovedEmbeddingEngine",
    "ImprovedVectorStore", 
    "EmbeddingConfig",
    
    # Модели данных
    "VectorDocument",
    "SearchResult",
    "SearchQuery",
    "BatchSearchQuery", 
    "HybridSearchQuery",
    "VectorStoreStats",
    "EmbeddingStats",
    "PerformanceMetrics",
    
    # Перечисления
    "SimilarityMetric",
    "PoolingStrategy",
    
    # Утилиты
    "benchmark_performance",
    "compare_models", 
    "stress_test",
    "optimize_store_config"
]