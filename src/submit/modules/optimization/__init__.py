"""
Модуль оптимизации производительности
"""

from .cache_manager import CacheManager, CacheEntry
from .batch_processor import BatchProcessor, EmbeddingBatchProcessor, FactExtractionBatchProcessor

__all__ = [
    'CacheManager',
    'CacheEntry',
    'BatchProcessor',
    'EmbeddingBatchProcessor', 
    'FactExtractionBatchProcessor'
]
