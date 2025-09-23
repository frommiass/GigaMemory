"""
Конфигурация для RAG системы с поддержкой сжатия
"""
from dataclasses import dataclass
from ..compression.compression_models import CompressionLevel, CompressionMethod


@dataclass
class RAGConfig:
    """Основные настройки RAG системы"""
    
    # Настройки классификации
    classification_confidence_threshold: float = 0.1
    
    # Настройки фильтрации
    max_relevant_sessions: int = 5
    min_relevant_sessions: int = 1
    
    # Настройки очистки сообщений
    min_message_length: int = 10
    max_message_length: int = 1000
    
    # Настройки ранжирования
    keyword_weight: float = 1.0
    recency_weight: float = 0.5
    session_length_weight: float = 0.3


@dataclass
class CompressedRAGConfig(RAGConfig):
    """Расширенная конфигурация для RAG с семантическим сжатием"""
    
    # Настройки компрессии
    enable_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    compression_method: CompressionMethod = CompressionMethod.HYBRID
    compression_target_ratio: float = 0.3
    compression_min_length: int = 200
    compression_max_length: int = 500
    
    # Иерархическая компрессия
    enable_hierarchical_compression: bool = True
    hierarchical_threshold: int = 10  # Количество сессий для включения иерархии
    
    # Кэширование сжатия
    compression_cache_enabled: bool = True
    compression_cache_size: int = 1000


# Глобальные конфигурации
DEFAULT_CONFIG = RAGConfig()
DEFAULT_COMPRESSED_CONFIG = CompressedRAGConfig()
