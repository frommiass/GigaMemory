# src/submit/model_inference.py
"""
Главный класс для конкурса - реализация ModelWithMemory
ИСПРАВЛЕННАЯ ВЕРСИЯ с правильной интеграцией
"""

import sys
import os
import logging
from typing import List

# Добавляем пути для импортов
sys.path.append(os.path.dirname(__file__) + '/../../')
from models import Message
from submit_interface import ModelWithMemory

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        
        # Создаем оптимизированную конфигурацию для конкурса
        config = {
            'model_path': model_path,
            
            'optimization': {
                'l1_cache_size': 200,      # Увеличили для критических данных
                'l2_cache_size': 5000,     # Оптимально для памяти
                'l2_cache_memory': 512,    # Экономим память
                'batch_size': 16,
                'num_workers': 2,          # Меньше воркеров для стабильности
                'default_ttl': 1800,
                'eviction_strategy': 'lru'
            },
            
            'storage': {
                'filter_copypaste': True,
                'cache_size': 5000,
                'min_message_length': 10,
                'max_message_length': 5000  # Ограничиваем для скорости
            },
            
            'embeddings': {
                'model_name': 'cointegrated/rubert-tiny2',
                'device': 'cuda',
                'batch_size': 32,
                'use_faiss': False,  # Отключаем FAISS если нет библиотеки
                'metric': 'cosine'
            },
            
            'extraction': {
                'min_confidence': 0.6,
                'use_llm': False,    # Только правила для скорости
                'use_rules': True,
                'conflict_strategy': 'latest',
                'filter_copypaste': True,
                'max_facts_per_session': 20  # Ограничиваем
            },
            
            'compression': {
                'level': 'moderate',
                'method': 'hybrid',
                'use_cache': True,
                'min_text_length': 300
            },
            
            'rag': {
                'top_k': 5,
                'use_hybrid_search': True,
                'use_compression': True,
                'max_context_length': 1500,  # Ограничиваем контекст
                'use_hierarchical': False
            }
        }
        
        # Инициализируем систему через bootstrap
        try:
            from .bootstrap import bootstrap_system_with_config, warmup_system
            self.orchestrator = bootstrap_system_with_config(config)
            
            # Прогреваем систему (опционально)
            try:
                warmup_system(self.orchestrator)
            except:
                pass  # Не критично если прогрев не удался
            
            logger.info("✅ System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            # Создаем fallback оркестратор
            self._create_fallback_orchestrator(model_path)
    
    def _create_fallback_orchestrator(self, model_path: str):
        """Создает минимальный оркестратор для fallback"""
        logger.warning("Creating fallback orchestrator...")
        
        try:
            # Минимальная инициализация
            from .core.container import container
            from .core.interfaces import IOptimizer, IStorage, IModelInference
            from .core.orchestrator import MemoryOrchestrator
            from .modules.optimization.module import OptimizationModule
            from .modules.storage.module import StorageModule
            from .llm_inference import ModelInference
            
            # Только критические модули
            optimizer = OptimizationModule({'l1_cache_size': 50})
            container.register_singleton(IOptimizer, optimizer)
            
            storage = StorageModule({})
            container.register_singleton(IStorage, storage)
            
            model = ModelInference(model_path)
            container.register_singleton(IModelInference, model)
            
            # Заглушки для остальных
            container.register_singleton(IEmbeddingEngine, None)
            container.register_singleton(IFactExtractor, None)
            container.register_singleton(ICompressor, None)
            container.register_singleton(IRAGEngine, None)
            
            self.orchestrator = MemoryOrchestrator()
            logger.warning("Fallback orchestrator created (limited functionality)")
            
        except Exception as e:
            logger.error(f"Failed to create fallback: {e}")
            raise RuntimeError("Cannot initialize system")
    
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Записывает сообщения в память
        
        Args:
            messages: Список сообщений для запоминания  
            dialogue_id: Идентификатор диалога
        """
        try:
            logger.info(f"Writing {len(messages)} messages to dialogue {dialogue_id}")
            
            # Валидация входных данных
            if not messages:
                logger.warning(f"Empty messages list for dialogue {dialogue_id}")
                return
            
            if not dialogue_id:
                logger.error("Empty dialogue_id provided")
                return
            
            # Обрабатываем диалог через оркестратор
            result = self.orchestrator.process_dialogue(dialogue_id, messages)
            
            if result.get('success', False):
                logger.info(
                    f"✅ Processed dialogue {dialogue_id}: "
                    f"{result.get('messages_processed', 0)} messages, "
                    f"{result.get('pipeline_results', {}).get('facts', {}).get('extracted', 0)} facts"
                )
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"❌ Failed to process dialogue {dialogue_id}: {error}")
                # НЕ выбрасываем исключение чтобы не сломать тестирование
            
        except Exception as e:
            logger.error(f"Error in write_to_memory: {e}", exc_info=True)
            # Продолжаем работу даже при ошибке
    
    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память для диалога
        
        Args:
            dialogue_id: Идентификатор диалога
        """
        try:
            logger.info(f"Clearing memory for dialogue {dialogue_id}")
            
            if not dialogue_id:
                logger.error("Empty dialogue_id for clear_memory")
                return
            
            self.orchestrator.clear_dialogue(dialogue_id)
            logger.info(f"✅ Memory cleared for dialogue {dialogue_id}")
            
        except Exception as e:
            logger.error(f"Error in clear_memory: {e}")
            # Не падаем, продолжаем работу
    
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
            logger.info(f"Answering question for dialogue {dialogue_id}: {question[:50]}...")
            
            # Валидация
            if not dialogue_id or not question:
                logger.warning("Empty dialogue_id or question")
                return "Нет такой информации."
            
            # Обрезаем слишком длинный вопрос
            if len(question) > 500:
                question = question[:500]
            
            # Получаем ответ через оркестратор
            answer = self.orchestrator.answer_question(dialogue_id, question)
            
            # Постобработка ответа
            answer = self._postprocess_answer(answer, question)
            
            logger.info(f"✅ Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error in answer_to_question: {e}", exc_info=True)
            return "Нет такой информации."
    
    def _postprocess_answer(self, answer: str, question: str) -> str:
        """
        Постобработка ответа для соответствия требованиям конкурса
        
        Args:
            answer: Исходный ответ
            question: Вопрос пользователя
            
        Returns:
            Обработанный ответ
        """
        # Убираем пустоту
        if not answer or not answer.strip():
            return "Нет такой информации."
        
        answer = answer.strip()
        
        # Убираем технические префиксы если есть
        prefixes_to_remove = [
            "Ответ:", "Answer:", "Ответ на вопрос:",
            "На основе контекста:", "Согласно диалогу:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Проверяем на "не знаю" ответы
        negative_patterns = [
            'не знаю', 'неизвестно', 'нет данных',
            'нет информации', 'не могу', 'не указано',
            'не упоминается', 'не сказано'
        ]
        
        answer_lower = answer.lower()
        if any(pattern in answer_lower for pattern in negative_patterns):
            return "Нет такой информации."
        
        # Ограничиваем длину (требование конкурса - 1 предложение)
        if len(answer) > 200:
            # Пытаемся взять первое предложение
            sentences = answer.split('.')
            if sentences and sentences[0]:
                answer = sentences[0].strip() + '.'
            else:
                # Обрезаем по символам
                answer = answer[:197] + "..."
        
        # Убеждаемся что ответ заканчивается точкой
        if answer and not answer[-1] in '.!?':
            answer += '.'
        
        # Финальная проверка длины
        if len(answer) > 200:
            answer = answer[:197] + "..."
        
        return answer

# Вспомогательная функция для тестирования
def test_system():
    """Функция для быстрого тестирования системы"""
    logger.info("Running system test...")
    
    try:
        # Создаем экземпляр
        model_path = "/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16"
        memory = SubmitModelWithMemory(model_path)
        
        # Тестовые сообщения
        test_messages = [
            Message("user", "Привет, меня зовут Иван."),
            Message("assistant", "Здравствуйте, Иван! Рад познакомиться."),
            Message("user", "Мне 25 лет и я работаю программистом."),
            Message("assistant", "Отлично! Программирование - интересная профессия."),
            Message("user", "У меня есть кот по имени Барсик."),
            Message("assistant", "Милое имя для кота!")
        ]
        
        # Записываем в память
        memory.write_to_memory(test_messages, "test_dialogue")
        
        # Тестируем вопросы
        questions = [
            "Как меня зовут?",
            "Сколько мне лет?",
            "Кем я работаю?",
            "Как зовут моего кота?",
            "Есть ли у меня собака?"
        ]
        
        print("\n=== Результаты теста ===")
        for q in questions:
            answer = memory.answer_to_question("test_dialogue", q)
            print(f"Q: {q}")
            print(f"A: {answer}\n")
        
        # Очищаем
        memory.clear_memory("test_dialogue")
        
        logger.info("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

# Для запуска теста
if __name__ == "__main__":
    test_system()