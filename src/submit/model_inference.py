# src/submit/model_inference.py

from typing import List
from models import Message
from submit_interface import ModelWithMemory
from bootstrap import bootstrap_system

class SubmitModelWithMemory(ModelWithMemory):
    """Реализация для соревнования"""
    
    def __init__(self, model_path: str):
        # Инициализируем систему
        config = {
            'model_path': model_path,
            'storage': {'cache_size': 10000, 'filter_copypaste': True},
            'embeddings': {'model_name': 'cointegrated/rubert-tiny2'},
            'extraction': {'min_confidence': 0.6},
            'compression': {'level': 'moderate'},
            'rag': {'top_k': 5}
        }
        
        self.orchestrator = bootstrap_system_with_config(config)
    
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """Записывает в память"""
        result = self.orchestrator.process_dialogue(dialogue_id, messages)
        if not result['success']:
            raise RuntimeError(f"Failed to process dialogue: {result.get('error')}")
    
    def clear_memory(self, dialogue_id: str) -> None:
        """Очищает память"""
        self.orchestrator.storage.clear(dialogue_id)
    
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос"""
        return self.orchestrator.answer_question(dialogue_id, question)

def bootstrap_system_with_config(config: dict):
    """Вспомогательная функция для инициализации"""
    from core.container import container
    from core.interfaces import *
    from core.orchestrator import MemoryOrchestrator
    
    # ... регистрация модулей ...
    
    return MemoryOrchestrator()