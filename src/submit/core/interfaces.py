# src/submit/core/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from models import Message
import numpy as np


# ============== БАЗОВЫЕ МОДЕЛИ ==============

@dataclass
class ProcessingResult:
    """Универсальный результат обработки"""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SessionData:
    """Данные сессии"""
    session_id: str
    dialogue_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = None


# ============== STORAGE INTERFACE ==============

class IStorage(ABC):
    """Интерфейс модуля хранилища"""
    
    @abstractmethod
    def store_messages(self, dialogue_id: str, messages: List[Message]) -> ProcessingResult:
        """Сохраняет сообщения с фильтрацией"""
        pass
    
    @abstractmethod
    def get_dialogue_messages(self, dialogue_id: str) -> ProcessingResult:
        """Получает все сообщения диалога"""
        pass
    
    @abstractmethod
    def get_dialogue_sessions(self, dialogue_id: str) -> ProcessingResult:
        """Получает сессии диалога"""
        pass
    
    @abstractmethod
    def clear_dialogue(self, dialogue_id: str) -> ProcessingResult:
        """Очищает данные диалога"""
        pass


# ============== EMBEDDINGS INTERFACE ==============

class IEmbeddingEngine(ABC):
    """Интерфейс модуля эмбеддингов"""
    
    @abstractmethod
    def encode_texts(self, texts: List[str]) -> ProcessingResult:
        """Создает эмбеддинги для текстов"""
        pass
    
    @abstractmethod
    def vector_search(self, query: str, dialogue_id: str, top_k: int = 5) -> ProcessingResult:
        """Векторный поиск по диалогу"""
        pass
    
    @abstractmethod
    def index_dialogue(self, dialogue_id: str, sessions: Dict[str, List[Message]]) -> ProcessingResult:
        """Индексирует диалог для поиска"""
        pass


# ============== EXTRACTION INTERFACE ==============

class IFactExtractor(ABC):
    """Интерфейс модуля извлечения фактов"""
    
    @abstractmethod
    def extract_facts(self, text: str, context: Dict[str, Any]) -> ProcessingResult:
        """Извлекает факты из текста"""
        pass
    
    @abstractmethod
    def get_user_profile(self, dialogue_id: str) -> ProcessingResult:
        """Создает профиль пользователя на основе фактов"""
        pass
    
    @abstractmethod
    def query_facts(self, dialogue_id: str, query: str) -> ProcessingResult:
        """Поиск фактов по запросу"""
        pass


# ============== COMPRESSION INTERFACE ==============

class ICompressor(ABC):
    """Интерфейс модуля сжатия"""
    
    @abstractmethod
    def compress_text(self, text: str, level: str = "moderate") -> ProcessingResult:
        """Сжимает текст"""
        pass
    
    @abstractmethod
    def compress_sessions(self, sessions: Dict[str, str]) -> ProcessingResult:
        """Сжимает множество сессий"""
        pass
    
    @abstractmethod
    def get_compression_stats(self) -> ProcessingResult:
        """Получает статистику сжатия"""
        pass


# ============== OPTIMIZATION INTERFACE ==============

class IOptimizer(ABC):
    """Интерфейс модуля оптимизации"""
    
    @abstractmethod
    def cache_get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша"""
        pass
    
    @abstractmethod
    def cache_put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Добавляет значение в кэш"""
        pass
    
    @abstractmethod
    def batch_process(self, tasks: List[Dict], processor_func: callable) -> ProcessingResult:
        """Батчевая обработка задач"""
        pass


# ============== RAG INTERFACE ==============

class IRAGEngine(ABC):
    """Интерфейс RAG модуля"""
    
    @abstractmethod
    def process_question(self, question: str, dialogue_id: str) -> ProcessingResult:
        """Обрабатывает вопрос и генерирует промпт"""
        pass
    
    @abstractmethod
    def find_relevant_sessions(self, question: str, dialogue_id: str) -> ProcessingResult:
        """Находит релевантные сессии"""
        pass
    
    @abstractmethod
    def generate_answer(self, question: str, context: str) -> ProcessingResult:
        """Генерирует ответ на основе контекста"""
        pass


# ============== MODEL INFERENCE INTERFACE ==============

class IModelInference(ABC):
    """Интерфейс для инференса модели"""
    
    @abstractmethod
    def generate(self, messages: List[Message]) -> ProcessingResult:
        """Генерирует ответ модели"""
        pass