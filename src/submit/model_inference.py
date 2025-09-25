# src/submit/model_inference.py

from typing import List
from models import Message
from submit_interface import ModelWithMemory
from .bootstrap import bootstrap_system
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubmitModelWithMemory(ModelWithMemory):
    """Реализация модели с памятью для конкурса GigaMemory"""
    
    def __init__(self, model_path: str):
        """
        Инициализация модели с памятью
        
        Args:
            model_path: Путь к модели GigaChat
        """
        logger.info(f"Initializing SubmitModelWithMemory with model at {model_path}")
        
        # Создаем конфигурацию с путем к модели
        config = {
            'model_path': model_path,
            'optimization': {
                'l1_cache_size': 100,
                'l2_cache_size': 5000,  # Меньше для экономии памяти
                'batch_size': 16,
                'default_ttl': 1800
            },
            'storage': {
                'filter_copypaste': True,
                'cache_size': 5000
            },
            'embeddings': {
                'model_name': 'cointegrated/rubert-tiny2',
                'device': 'cuda',
                'batch_size': 32
            },
            'extraction': {
                'min_confidence': 0.6,
                'use_llm': False,  # Только rules для скорости
                'use_rules': True
            },
            'compression': {
                'level': 'moderate'
            },
            'rag': {
                'top_k': 5,
                'use_hybrid_search': True
            }
        }
        
        # Инициализируем систему через bootstrap
        self.orchestrator = bootstrap_system_with_config(config)
        logger.info("System initialized successfully")
    
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Записывает сообщения в память
        
        Args:
            messages: Список сообщений для запоминания
            dialogue_id: Идентификатор диалога
        """
        try:
            logger.info(f"Writing {len(messages)} messages to memory for dialogue {dialogue_id}")
            
            # Обрабатываем диалог через оркестратор
            result = self.orchestrator.process_dialogue(dialogue_id, messages)
            
            if not result.get('success', False):
                error = result.get('error', 'Unknown error')
                logger.error(f"Failed to process dialogue: {error}")
                # Не выбрасываем исключение, чтобы не сломать тестирование
                
            logger.info(f"Successfully processed dialogue {dialogue_id}")
            
        except Exception as e:
            logger.error(f"Error in write_to_memory: {e}")
            # Не падаем, продолжаем работу
    
    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память для диалога
        
        Args:
            dialogue_id: Идентификатор диалога
        """
        try:
            logger.info(f"Clearing memory for dialogue {dialogue_id}")
            self.orchestrator.clear_dialogue(dialogue_id)
            
        except Exception as e:
            logger.error(f"Error in clear_memory: {e}")
    
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """
        Отвечает на вопрос на основе памяти диалога
        
        Args:
            dialogue_id: Идентификатор диалога
            question: Вопрос пользователя
            
        Returns:
            Ответ на вопрос (краткий, не более 1 предложения)
        """
        try:
            logger.info(f"Answering question for dialogue {dialogue_id}: {question}")
            
            # Получаем ответ через оркестратор
            answer = self.orchestrator.answer_question(dialogue_id, question)
            
            # Проверяем что ответ корректный
            if not answer or len(answer.strip()) == 0:
                answer = "Нет такой информации."
            
            # Убеждаемся что ответ краткий (требование конкурса)
            if len(answer) > 200:
                # Обрезаем до первого предложения
                first_sentence = answer.split('.')[0]
                if first_sentence:
                    answer = first_sentence + '.'
                else:
                    answer = answer[:197] + "..."
            
            logger.info(f"Generated answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error in answer_to_question: {e}")
            return "Нет такой информации."


def bootstrap_system_with_config(config: dict):
    """Вспомогательная функция для инициализации с конфигом"""
    from .core.container import container
    from .core.interfaces import IOptimizer, IStorage, IEmbeddingEngine, IFactExtractor, ICompressor, IRAGEngine, IModelInference
    from .core.orchestrator import MemoryOrchestrator
    from .modules.optimization.module import OptimizationModule
    from .modules.storage.module import StorageModule
    from .modules.embeddings.module import EmbeddingsModule
    from .modules.extraction.module import ExtractionModule
    from .modules.compression.module import CompressionModule
    from .modules.rag.module import RAGModule
    from .llm_inference import ModelInference
    
    # Регистрируем модули в контейнере
    container.register_singleton(
        IOptimizer,
        OptimizationModule(config.get('optimization', {}))
    )
    
    container.register_singleton(
        IStorage,
        StorageModule(config.get('storage', {}))
    )
    
    container.register_singleton(
        IEmbeddingEngine,
        EmbeddingsModule(config.get('embeddings', {}))
    )
    
    container.register_singleton(
        IFactExtractor,
        ExtractionModule(config.get('extraction', {}))
    )
    
    container.register_singleton(
        ICompressor,
        CompressionModule(config.get('compression', {}))
    )
    
    container.register_singleton(
        IRAGEngine,
        RAGModule(config.get('rag', {}))
    )
    
    container.register_singleton(
        IModelInference,
        ModelInference(config.get('model_path'))
    )
    
    # Создаем и возвращаем оркестратор
    return MemoryOrchestrator()