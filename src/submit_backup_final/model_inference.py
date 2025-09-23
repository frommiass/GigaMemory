from typing import List
from models import Message
from submit_interface import ModelWithMemory

from .storage import MemoryStorage
from .rag.engine import RAGEngine
from .llm_inference import ModelInference


class SubmitModelWithMemory(ModelWithMemory):
    """
    Система памяти с RAG для обработки вопросов пользователя
    """

    def __init__(self, model_path: str) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        self.rag_interface = RAGEngine()

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Фильтрует, сохраняет и группирует личную информацию из сообщений
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

    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память диалога
        """
        self.storage.clear_dialogue_memory(dialogue_id)
        
        if self.storage.get_cache_size() > 1000:
            self.storage.clear_all_cache()

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """
        Генерирует ответ на вопрос используя RAG систему
        """
        # Получаем все сообщения из памяти
        memory = self.storage.get_memory(dialogue_id)
        
        if not memory:
            return "У меня нет информации для ответа на этот вопрос."
        
        # Используем RAG систему для генерации промпта
        rag_prompt, metadata = self.rag_interface.process_question(question, dialogue_id, memory)
        
        # Создаем контекст для модели
        context_with_memory = [Message('system', rag_prompt)]
        
        # Генерируем ответ через модель
        answer = self.model_inference.inference(context_with_memory)
        return answer