"""
Конфигурация для RAG системы
"""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Основные настройки RAG системы"""
    
    # Настройки классификации
    classification_confidence_threshold: float = 0.7
    
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


# Глобальная конфигурация
DEFAULT_CONFIG = RAGConfig()
