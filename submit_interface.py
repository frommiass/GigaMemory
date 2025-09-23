"""
Интерфейс для системы GigaMemory
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from models import Message, Dialogue


class ModelWithMemory(ABC):
    """Базовый класс для модели с памятью"""
    
    @abstractmethod
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """Записать сообщения в память"""
        pass
    
    @abstractmethod
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """Ответить на вопрос используя память"""
        pass
    
    @abstractmethod
    def clear_memory(self, dialogue_id: str) -> None:
        """Очистить память диалога"""
        pass


class ModelInference(ABC):
    """Базовый класс для инференса модели"""
    
    @abstractmethod
    def inference(self, messages: List[Message]) -> str:
        """Генерация ответа модели"""
        pass
