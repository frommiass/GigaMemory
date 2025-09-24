# src/submit/core/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from models import Message

@dataclass
class ProcessingResult:
    """Результат обработки"""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None

class IStorage(ABC):
    """Интерфейс для хранилища"""
    @abstractmethod
    def add_messages(self, dialogue_id: str, messages: List[Message]) -> ProcessingResult:
        pass
    
    @abstractmethod
    def get_messages(self, dialogue_id: str) -> ProcessingResult:
        pass
    
    @abstractmethod
    def clear(self, dialogue_id: str) -> ProcessingResult:
        pass

class IEmbeddingEngine(ABC):
    """Интерфейс для эмбеддингов"""
    @abstractmethod
    def encode(self, texts: List[str]) -> ProcessingResult:
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> ProcessingResult:
        pass

class IFactExtractor(ABC):
    """Интерфейс для извлечения фактов"""
    @abstractmethod
    def extract_facts(self, text: str, context: Dict) -> ProcessingResult:
        pass

class ICompressor(ABC):
    """Интерфейс для сжатия"""
    @abstractmethod
    def compress(self, text: str, level: str = "moderate") -> ProcessingResult:
        pass

class IRAGEngine(ABC):
    """Интерфейс для RAG"""
    @abstractmethod
    def process_question(self, question: str, context: Dict) -> ProcessingResult:
        pass

class IModelInference(ABC):
    """Интерфейс для инференса модели"""
    @abstractmethod
    def generate(self, messages: List[Message]) -> ProcessingResult:
        pass