from typing import List, Dict, Any, Optional
from models import Message
from submit_interface import ModelWithMemory

from .storage import MemoryStorage
from .rag.vector_rag_interface import VectorRAGInterface
from .llm_inference import ModelInference


class SubmitModelWithVectorMemory(ModelWithMemory):
    """
    Система памяти с векторным RAG для обработки вопросов пользователя
    Использует улучшенный векторный поиск вместо традиционного RAG
    """

    def __init__(self, model_path: str, 
                 use_vector_rag: bool = True,
                 vector_model: str = "cointegrated/rubert-tiny2",
                 use_gpu: bool = True,
                 enable_hybrid_search: bool = True) -> None:
        """
        Инициализация с поддержкой векторного RAG
        
        Args:
            model_path: Путь к модели для генерации ответов
            use_vector_rag: Использовать векторный RAG (True) или обычный (False)
            vector_model: Модель для создания эмбеддингов
            use_gpu: Использовать GPU для эмбеддингов
            enable_hybrid_search: Включить гибридный поиск
        """
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        self.use_vector_rag = use_vector_rag
        
        if use_vector_rag:
            # Используем векторный RAG
            self.rag_interface = VectorRAGInterface(
                model_name=vector_model,
                use_gpu=use_gpu,
                enable_hybrid_search=enable_hybrid_search
            )
        else:
            # Используем обычный RAG (для обратной совместимости)
            from .rag.engine import RAGEngine
            self.rag_interface = RAGEngine()

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Фильтрует, сохраняет и группирует личную информацию из сообщений
        С автоматической векторизацией для векторного RAG
        """
        # Группируем сообщения по сессиям (используем существующую логику)
        session_messages = {}
        
        for msg in messages:
            if msg.role == "user":
                # Проверяем кэш для ускорения
                cached_result = self.storage.check_cache(msg.content)
                
                if cached_result is None:
                    from .core.message_filter import is_personal_message
                    filter_result = is_personal_message(msg.content)
                    self.storage.add_to_cache(msg.content, filter_result)
                    cached_result = filter_result
                
                if cached_result:
                    # Используем session_id из сообщения, если он есть
                    session_id = msg.session_id if msg.session_id else str(self.storage.increment_session(dialogue_id))
                    
                    # Регистрируем сессию
                    if msg.session_id:
                        self.storage.register_session(dialogue_id, session_id)
                    
                    # Группируем по session_id
                    if session_id not in session_messages:
                        session_messages[session_id] = []
                    
                    session_messages[session_id].append(msg)
        
        # Сохраняем сообщения в память (БЕЗ предварительного склеивания)
        processed_messages = []
        for session_id, msgs in session_messages.items():
            for msg in msgs:
                processed_msg = Message(
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id
                )
                processed_messages.append(processed_msg)
        
        self.storage.add_to_memory(dialogue_id, processed_messages)
        
        # Если используем векторный RAG, индексируем сообщения
        if self.use_vector_rag and processed_messages:
            self._index_messages_for_vector_rag(dialogue_id, processed_messages)

    def _index_messages_for_vector_rag(self, dialogue_id: str, messages: List[Message]) -> None:
        """
        Индексирует сообщения для векторного RAG
        
        Args:
            dialogue_id: ID диалога
            messages: Список сообщений для индексации
        """
        try:
            # Конвертируем в формат для VectorRAGInterface
            formatted_messages = [
                {"role": msg.role, "content": msg.content} 
                for msg in messages
            ]
            
            # Добавляем в векторный RAG (автоматическая векторизация)
            self.rag_interface.add_dialogue(dialogue_id, formatted_messages)
            
        except Exception as e:
            # Логируем ошибку но не прерываем выполнение
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Ошибка индексации для векторного RAG: {e}")

    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память диалога
        """
        self.storage.clear_dialogue_memory(dialogue_id)
        
        # Очищаем векторные индексы если используем векторный RAG
        if self.use_vector_rag and hasattr(self.rag_interface, 'stores'):
            if dialogue_id in self.rag_interface.stores:
                del self.rag_interface.stores[dialogue_id]
            if dialogue_id in self.rag_interface.dialogues:
                del self.rag_interface.dialogues[dialogue_id]
        
        if self.storage.get_cache_size() > 1000:
            self.storage.clear_all_cache()

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """
        Генерирует ответ на вопрос используя векторный RAG
        """
        # Получаем все сообщения из памяти
        memory = self.storage.get_memory(dialogue_id)
        
        if not memory:
            return "У меня нет информации для ответа на этот вопрос."
        
        if self.use_vector_rag:
            # Используем векторный RAG
            return self._answer_with_vector_rag(dialogue_id, question, memory)
        else:
            # Используем обычный RAG (для обратной совместимости)
            return self._answer_with_traditional_rag(dialogue_id, question, memory)

    def _answer_with_vector_rag(self, dialogue_id: str, question: str, memory: List[Message]) -> str:
        """
        Ответ с использованием векторного RAG
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            memory: Сообщения из памяти
            
        Returns:
            Ответ на вопрос
        """
        try:
            # Получаем релевантный контекст через векторный поиск
            context = self.rag_interface.get_relevant_context(
                question, 
                dialogue_id,
                top_k=5  # Топ-5 релевантных фрагментов
            )
            
            # Формируем промпт с контекстом
            if context:
                enhanced_prompt = f"{context}\n\nВопрос: {question}"
            else:
                # Если векторный поиск не дал результатов, используем традиционный RAG
                enhanced_prompt = self._get_fallback_context(question, memory)
            
            # Создаем контекст для модели
            context_with_memory = [Message('system', enhanced_prompt)]
            
            # Генерируем ответ через модель
            answer = self.model_inference.inference(context_with_memory)
            
            # Индексируем ответ для будущего использования
            self.rag_interface.process_message(
                answer,
                dialogue_id,
                role="assistant"
            )
            
            return answer
            
        except Exception as e:
            # В случае ошибки используем традиционный RAG
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Ошибка векторного RAG, переключаемся на традиционный: {e}")
            return self._answer_with_traditional_rag(dialogue_id, question, memory)

    def _answer_with_traditional_rag(self, dialogue_id: str, question: str, memory: List[Message]) -> str:
        """
        Ответ с использованием традиционного RAG (для обратной совместимости)
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            memory: Сообщения из памяти
            
        Returns:
            Ответ на вопрос
        """
        # Используем традиционный RAG
        rag_prompt, metadata = self.rag_interface.process_question(question, dialogue_id, memory)
        
        # Создаем контекст для модели
        context_with_memory = [Message('system', rag_prompt)]
        
        # Генерируем ответ через модель
        answer = self.model_inference.inference(context_with_memory)
        return answer

    def _get_fallback_context(self, question: str, memory: List[Message]) -> str:
        """
        Получает контекст через традиционный RAG как fallback
        
        Args:
            question: Вопрос пользователя
            memory: Сообщения из памяти
            
        Returns:
            Контекст для промпта
        """
        try:
            # Создаем временный традиционный RAG для fallback
            from .rag.engine import RAGEngine
            fallback_rag = RAGEngine()
            
            # Получаем промпт через традиционный RAG
            rag_prompt, _ = fallback_rag.process_question(question, "fallback", memory)
            return rag_prompt
            
        except Exception:
            # Если и это не работает, возвращаем простой промпт
            return f"Контекст из истории диалога:\n{self._format_memory_simple(memory)}\n\nВопрос: {question}"

    def _format_memory_simple(self, memory: List[Message]) -> str:
        """
        Простое форматирование памяти
        
        Args:
            memory: Список сообщений
            
        Returns:
            Отформатированная строка
        """
        if not memory:
            return "История диалога пуста."
        
        formatted = []
        for msg in memory[-10:]:  # Последние 10 сообщений
            formatted.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(formatted)

    def get_vector_search_results(self, question: str, dialogue_id: str, top_k: int = 5) -> List[Dict]:
        """
        Получает результаты векторного поиска для анализа
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            top_k: Количество результатов
            
        Returns:
            Список результатов поиска
        """
        if not self.use_vector_rag:
            return []
        
        try:
            return self.rag_interface.search(question, dialogue_id, top_k=top_k)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Ошибка векторного поиска: {e}")
            return []

    def get_vector_rag_stats(self) -> Dict[str, Any]:
        """
        Получает статистику векторного RAG
        
        Returns:
            Словарь со статистикой
        """
        if not self.use_vector_rag:
            return {"vector_rag_enabled": False}
        
        try:
            stats = {
                "vector_rag_enabled": True,
                "embedding_stats": self.rag_interface.embedding_engine.get_stats(),
                "total_dialogues": len(self.rag_interface.dialogues),
                "total_stores": len(self.rag_interface.stores)
            }
            
            # Добавляем статистику по хранилищам
            store_stats = {}
            for dialogue_id, store in self.rag_interface.stores.items():
                analytics = store.get_analytics()
                store_stats[dialogue_id] = {
                    "documents": analytics.total_documents,
                    "searches": analytics.total_searches,
                    "avg_search_time": analytics.avg_search_time
                }
            
            stats["store_stats"] = store_stats
            return stats
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Ошибка получения статистики векторного RAG: {e}")
            return {"vector_rag_enabled": True, "error": str(e)}

    def save_vector_indices(self, path: str = "./vector_indices") -> None:
        """
        Сохраняет векторные индексы
        
        Args:
            path: Путь для сохранения
        """
        if self.use_vector_rag:
            try:
                self.rag_interface.save(path)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Ошибка сохранения векторных индексов: {e}")

    def load_vector_indices(self, path: str = "./vector_indices") -> None:
        """
        Загружает векторные индексы
        
        Args:
            path: Путь для загрузки
        """
        if self.use_vector_rag:
            try:
                self.rag_interface.load(path)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Ошибка загрузки векторных индексов: {e}")


# Для обратной совместимости
class SubmitModelWithMemory(SubmitModelWithVectorMemory):
    """
    Обратная совместимость с оригинальным SubmitModelWithMemory
    По умолчанию использует векторный RAG
    """
    
    def __init__(self, model_path: str) -> None:
        super().__init__(
            model_path=model_path,
            use_vector_rag=True,  # По умолчанию векторный RAG
            vector_model="cointegrated/rubert-tiny2",
            use_gpu=True,
            enable_hybrid_search=True
        )
