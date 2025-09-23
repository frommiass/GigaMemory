"""
Модели данных для векторного поиска
"""
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SimilarityMetric(Enum):
    """Метрики сходства"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"
    ANGULAR = "angular"  # Угловое расстояние


class PoolingStrategy(Enum):
    """Стратегии пулинга для эмбеддингов"""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    MEAN_MAX = "mean_max"  # Конкатенация mean и max
    WEIGHTED_MEAN = "weighted_mean"  # С весами по важности токенов


@dataclass
class EmbeddingConfig:
    """Расширенная конфигурация для движка эмбеддингов"""
    model_name: str = "cointegrated/rubert-tiny2"
    device: str = "cuda"
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    cache_dir: Optional[str] = None
    use_cache: bool = True
    
    # Новые параметры
    use_amp: bool = True  # Automatic Mixed Precision для ускорения
    num_workers: int = 4  # Для параллельной обработки
    prefetch_batches: int = 2  # Предзагрузка батчей
    warmup_steps: int = 0  # Прогрев модели
    compile_model: bool = False  # torch.compile для ускорения (PyTorch 2.0+)
    quantize_model: bool = False  # Квантизация для уменьшения памяти


@dataclass
class VectorDocument:
    """Расширенный документ в векторном хранилище"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class SearchResult:
    """Расширенный результат поиска"""
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None
    rank: int = 0
    explanation: Optional[Dict[str, Any]] = None  # Объяснение релевантности


@dataclass
class SearchQuery:
    """Запрос для поиска"""
    vector: np.ndarray
    text: Optional[str] = None
    k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    filter_text: Optional[str] = None
    threshold: Optional[float] = None
    rerank: bool = False
    return_explanations: bool = False


@dataclass
class BatchSearchQuery:
    """Батчевый запрос для поиска"""
    vectors: np.ndarray
    texts: Optional[List[str]] = None
    k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    filter_text: Optional[str] = None
    threshold: Optional[float] = None
    rerank: bool = False
    return_explanations: bool = False


@dataclass
class HybridSearchQuery:
    """Гибридный запрос для поиска"""
    vector: np.ndarray
    text: str
    k: int = 5
    vector_weight: float = 0.7
    text_weight: float = 0.3
    filter_metadata: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    rerank: bool = False
    return_explanations: bool = False


@dataclass
class VectorStoreStats:
    """Статистика векторного хранилища"""
    total_documents: int = 0
    total_searches: int = 0
    avg_search_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    popular_queries: Dict[str, int] = field(default_factory=dict)
    access_patterns: Dict[str, int] = field(default_factory=dict)
    top_accessed_docs: List[Dict[str, Any]] = field(default_factory=list)
    unique_metadata_keys: List[str] = field(default_factory=list)


@dataclass
class EmbeddingStats:
    """Статистика движка эмбеддингов"""
    total_encoded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    encoding_time: float = 0.0
    last_batch_time: float = 0.0
    cache_size: int = 0
    cache_hit_rate: float = 0.0
    avg_encoding_speed: float = 0.0  # документов в секунду


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    encoding_speed: float = 0.0  # документов/сек
    search_speed: float = 0.0    # поисков/сек
    memory_efficiency: float = 0.0  # документов/МБ
    cache_efficiency: float = 0.0   # hit rate
    accuracy_score: float = 0.0     # точность поиска
    latency_p95: float = 0.0        # 95-й перцентиль задержки
    throughput: float = 0.0         # общая пропускная способность
