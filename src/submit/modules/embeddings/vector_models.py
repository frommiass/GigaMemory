# src/submit/modules/embeddings/vector_models.py
"""
Модели данных для векторного поиска
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SimilarityMetric(Enum):
    """Метрики сходства для векторного поиска"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"
    ANGULAR = "angular"


class PoolingStrategy(Enum):
    """Стратегии пулинга для эмбеддингов"""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    MEAN_MAX = "mean_max"
    WEIGHTED_MEAN = "weighted_mean"


@dataclass
class VectorDocument:
    """Документ с векторным представлением"""
    id: str
    vector: Any  # numpy array или list
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует документ в словарь"""
        return {
            'id': self.id,
            'metadata': self.metadata,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count
        }


@dataclass
class SearchResult:
    """Результат векторного поиска"""
    doc_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    rank: int = 0
    explanation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь"""
        return {
            'doc_id': self.doc_id,
            'score': self.score,
            'metadata': self.metadata,
            'text': self.text,
            'rank': self.rank,
            'explanation': self.explanation
        }


@dataclass
class SearchQuery:
    """Запрос для векторного поиска"""
    query_vector: Any  # numpy array
    top_k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    filter_text: Optional[str] = None
    threshold: Optional[float] = None
    rerank: bool = False
    return_explanations: bool = False


@dataclass
class BatchSearchQuery:
    """Батчевый запрос для векторного поиска"""
    query_vectors: List[Any]  # List of numpy arrays
    top_k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None


@dataclass
class HybridSearchQuery:
    """Гибридный запрос (векторный + текстовый)"""
    query_vector: Any
    query_text: str
    top_k: int = 5
    vector_weight: float = 0.7
    text_weight: float = 0.3
    filter_metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingConfig:
    """Конфигурация для движка эмбеддингов"""
    model_name: str = "cointegrated/rubert-tiny2"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    cache_dir: Optional[str] = None
    use_cache: bool = True
    use_amp: bool = False  # Automatic Mixed Precision
    num_workers: int = 1
    prefetch_batches: int = 2
    warmup_steps: int = 0
    compile_model: bool = False
    quantize_model: bool = False


@dataclass
class VectorStoreStats:
    """Статистика векторного хранилища"""
    total_documents: int = 0
    total_searches: int = 0
    avg_search_time: float = 0.0
    unique_metadata_keys: List[str] = field(default_factory=list)
    avg_access_count: float = 0.0
    max_access_count: int = 0
    top_accessed_docs: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует статистику в словарь"""
        return {
            'total_documents': self.total_documents,
            'total_searches': self.total_searches,
            'avg_search_time': self.avg_search_time,
            'unique_metadata_keys': self.unique_metadata_keys,
            'avg_access_count': self.avg_access_count,
            'max_access_count': self.max_access_count,
            'top_accessed_docs': self.top_accessed_docs
        }


@dataclass
class EmbeddingStats:
    """Статистика работы движка эмбеддингов"""
    total_encoded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    avg_encoding_time: float = 0.0
    model_name: str = ""
    device: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует статистику в словарь"""
        return {
            'total_encoded': self.total_encoded,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hit_rate,
            'avg_encoding_time': self.avg_encoding_time,
            'model_name': self.model_name,
            'device': self.device
        }


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    operation: str
    duration: float
    items_processed: int
    throughput: float = 0.0
    memory_used: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Вычисляет производительность"""
        if self.duration > 0:
            self.throughput = self.items_processed / self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метрики в словарь"""
        return {
            'operation': self.operation,
            'duration': self.duration,
            'items_processed': self.items_processed,
            'throughput': self.throughput,
            'memory_used': self.memory_used,
            'timestamp': self.timestamp.isoformat()
        }