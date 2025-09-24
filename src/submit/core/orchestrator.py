# src/submit/core/orchestrator.py

from typing import List, Dict, Any
from models import Message
from .interfaces import *
from .container import container
import logging

logger = logging.getLogger(__name__)

class MemoryOrchestrator:
    """Главный оркестратор системы памяти"""
    
    def __init__(self):
        self.storage = container.get(IStorage)
        self.embeddings = container.get(IEmbeddingEngine)
        self.extractor = container.get(IFactExtractor)
        self.compressor = container.get(ICompressor)
        self.rag = container.get(IRAGEngine)
        self.model = container.get(IModelInference)
    
    def process_dialogue(self, dialogue_id: str, messages: List[Message]) -> Dict[str, Any]:
        """Обрабатывает диалог через все компоненты"""
        
        pipeline_results = {}
        
        # 1. Сохранение и фильтрация
        storage_result = self.storage.add_messages(dialogue_id, messages)
        pipeline_results['storage'] = storage_result
        
        if not storage_result.success:
            return self._error_response(storage_result.error)
        
        filtered_messages = storage_result.data
        
        # 2. Извлечение фактов (параллельно)
        facts_results = []
        for msg in filtered_messages:
            if msg.role == "user":
                fact_result = self.extractor.extract_facts(
                    msg.content,
                    {'dialogue_id': dialogue_id, 'session_id': getattr(msg, 'session_id', '0')}
                )
                if fact_result.success:
                    facts_results.extend(fact_result.data)
        
        pipeline_results['facts'] = {
            'count': len(facts_results),
            'facts': facts_results
        }
        
        # 3. Векторизация (батчами)
        if filtered_messages:
            texts = [msg.content for msg in filtered_messages]
            embedding_result = self.embeddings.encode(texts)
            pipeline_results['embeddings'] = embedding_result
        
        # 4. Сжатие для больших диалогов
        total_length = sum(len(msg.content) for msg in filtered_messages)
        if total_length > 10000:
            compressed_texts = []
            for msg in filtered_messages:
                if len(msg.content) > 500:
                    compress_result = self.compressor.compress(msg.content)
                    if compress_result.success:
                        compressed_texts.append(compress_result.data)
                else:
                    compressed_texts.append(msg.content)
            
            pipeline_results['compression'] = {
                'original_length': total_length,
                'compressed_length': sum(len(t) for t in compressed_texts),
                'ratio': sum(len(t) for t in compressed_texts) / total_length
            }
        
        return {
            'success': True,
            'dialogue_id': dialogue_id,
            'pipeline_results': pipeline_results
        }
    
    def answer_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос используя все компоненты"""
        
        # 1. Получаем сохраненные сообщения
        storage_result = self.storage.get_messages(dialogue_id)
        
        if not storage_result.success or not storage_result.data:
            return "У меня нет информации для ответа на этот вопрос."
        
        # 2. RAG обработка
        rag_result = self.rag.process_question(
            question,
            {'dialogue_id': dialogue_id, 'messages': storage_result.data}
        )
        
        if not rag_result.success:
            return "Произошла ошибка при обработке вопроса."
        
        # 3. Генерация ответа
        prompt_message = Message('system', rag_result.data)
        model_result = self.model.generate([prompt_message])
        
        if model_result.success:
            return model_result.data
        else:
            return "Не удалось сгенерировать ответ."
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Формирует ответ об ошибке"""
        logger.error(f"Pipeline error: {error}")
        return {
            'success': False,
            'error': error
        }