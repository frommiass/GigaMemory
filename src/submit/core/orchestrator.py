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
        # Получаем все модули из контейнера
        self.optimizer = container.get(IOptimizer)
        self.storage = container.get(IStorage)
        self.embeddings = container.get(IEmbeddingEngine)
        self.extractor = container.get(IFactExtractor)
        self.compressor = container.get(ICompressor)
        self.rag = container.get(IRAGEngine)
        self.model = container.get(IModelInference)
        
        # КРИТИЧНО: Устанавливаем зависимости между модулями
        self._setup_dependencies()
        
        logger.info("MemoryOrchestrator initialized with all modules")
    
    def _setup_dependencies(self):
        """Устанавливает зависимости между модулями"""
        
        # Storage может использовать optimizer
        if hasattr(self.storage, 'set_dependencies'):
            self.storage.set_dependencies(optimizer=self.optimizer)
        
        # Embeddings может использовать optimizer и storage
        if hasattr(self.embeddings, 'set_dependencies'):
            self.embeddings.set_dependencies(
                optimizer=self.optimizer,
                storage=self.storage
            )
        
        # Extractor может использовать все
        if hasattr(self.extractor, 'set_dependencies'):
            self.extractor.set_dependencies(
                optimizer=self.optimizer,
                storage=self.storage,
                embeddings=self.embeddings
            )
        
        # Compressor может использовать optimizer
        if hasattr(self.compressor, 'set_dependencies'):
            self.compressor.set_dependencies(
                optimizer=self.optimizer
            )
        
        # RAG использует все модули
        if hasattr(self.rag, 'set_dependencies'):
            self.rag.set_dependencies(
                storage=self.storage,
                embeddings=self.embeddings,
                extractor=self.extractor,
                compressor=self.compressor,
                optimizer=self.optimizer
            )
        
        logger.info("Module dependencies configured")
    
    def process_dialogue(self, dialogue_id: str, messages: List[Message]) -> Dict[str, Any]:
        """Обрабатывает диалог через все компоненты"""
        
        pipeline_results = {}
        
        try:
            # 1. Сохранение и фильтрация
            storage_result = self.storage.store_messages(dialogue_id, messages)
            pipeline_results['storage'] = storage_result
            
            if not storage_result.success:
                return self._error_response(f"Storage failed: {storage_result.error}")
            
            filtered_messages = storage_result.data
            
            # 2. Получаем сессии для индексации
            sessions_result = self.storage.get_dialogue_sessions(dialogue_id)
            if sessions_result.success:
                sessions = sessions_result.data
                
                # 3. Индексируем для векторного поиска
                if self.embeddings:
                    index_result = self.embeddings.index_dialogue(dialogue_id, sessions)
                    pipeline_results['embeddings'] = index_result
                
                # 4. Извлекаем факты из каждой сессии
                if self.extractor:
                    facts_extracted = 0
                    for session_id, session_messages in sessions.items():
                        # Объединяем тексты сессии
                        session_text = '\n'.join([
                            msg.content for msg in session_messages 
                            if hasattr(msg, 'content')
                        ])
                        
                        if session_text:
                            fact_result = self.extractor.extract_facts(
                                session_text,
                                {'dialogue_id': dialogue_id, 'session_id': session_id}
                            )
                            if fact_result.success:
                                facts_extracted += len(fact_result.data)
                    
                    pipeline_results['facts'] = {'extracted': facts_extracted}
            
            # 5. Сжатие для больших диалогов (опционально)
            total_length = sum(len(msg.content) for msg in filtered_messages if hasattr(msg, 'content'))
            if total_length > 10000 and self.compressor:
                # Сжимаем длинные сессии
                compression_stats = {'compressed_sessions': 0, 'saved_chars': 0}
                pipeline_results['compression'] = compression_stats
            
            return {
                'success': True,
                'dialogue_id': dialogue_id,
                'messages_processed': len(filtered_messages),
                'pipeline_results': pipeline_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._error_response(str(e))
    
    def answer_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос используя все компоненты"""
        
        try:
            # 1. Обрабатываем вопрос через RAG
            rag_result = self.rag.process_question(question, dialogue_id)
            
            if not rag_result.success:
                # Fallback: пытаемся ответить без RAG
                return self._fallback_answer(dialogue_id, question)
            
            # 2. Получаем промпт из RAG
            prompt = rag_result.data
            
            # 3. Генерируем ответ через модель
            messages = [
                Message('system', 'Ты полезный ассистент. Отвечай на русском языке, кратко и точно, не более одного предложения.'),
                Message('user', prompt)
            ]
            
            model_result = self.model.generate(messages)
            
            if model_result.success:
                answer = model_result.data
                # Очищаем ответ
                answer = answer.strip()
                
                # Проверяем что ответ не пустой
                if not answer or answer.lower() in ['не знаю', 'неизвестно', 'нет данных']:
                    return "Нет такой информации."
                
                if len(answer) > 200:  # Ограничиваем длину для конкурса
                    answer = answer[:197] + "..."
                return answer
            else:
                return "Нет такой информации."
                
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Нет такой информации."
    
    def _fallback_answer(self, dialogue_id: str, question: str) -> str:
        """Резервный метод ответа без RAG"""
        try:
            # Пытаемся найти факты напрямую
            if self.extractor:
                facts_result = self.extractor.query_facts(dialogue_id, question)
                if facts_result.success and facts_result.data:
                    # Формируем ответ из фактов
                    facts = facts_result.data[:3]  # Топ-3 факта
                    if facts:
                        # Простой промпт с фактами
                        facts_text = '\n'.join([str(f) for f in facts])
                        prompt = f"""Используя следующие факты о пользователе:
{facts_text}

Ответь на вопрос: {question}
Ответ должен быть на русском языке, кратким и точным (одно предложение)."""
                        
                        messages = [
                            Message('system', 'Отвечай только на основе предоставленных фактов. Если информации недостаточно, скажи "Нет такой информации."'),
                            Message('user', prompt)
                        ]
                        model_result = self.model.generate(messages)
                        if model_result.success:
                            answer = model_result.data.strip()
                            if answer and not answer.lower().startswith('не'):
                                return answer
            
            return "Нет такой информации."
            
        except Exception as e:
            logger.error(f"Fallback answer error: {e}")
            return "Нет такой информации."
    
    def clear_dialogue(self, dialogue_id: str):
        """Очищает все данные диалога"""
        try:
            # Очищаем в каждом модуле
            if self.storage:
                self.storage.clear_dialogue(dialogue_id)
            
            # Другие модули могут иметь свои методы очистки
            # но они не обязательны по интерфейсу
            
            logger.info(f"Dialogue {dialogue_id} cleared")
            
        except Exception as e:
            logger.error(f"Error clearing dialogue: {e}")
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Формирует ответ об ошибке"""
        logger.error(f"Pipeline error: {error}")
        return {
            'success': False,
            'error': error
        }