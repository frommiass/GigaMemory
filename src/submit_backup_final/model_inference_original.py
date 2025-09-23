from dataclasses import asdict
from typing import List

from models import Message
from submit_interface import ModelWithMemory

from .storage import MemoryStorage
from .filters.message_cleaner import is_personal_message
from .prompts import get_session_marker_prompt
from .prompts.fallback_prompts import get_fallback_prompt
from .rag import RAGInterface
# from .llm_inference import ModelInference  # Временно закомментировано для тестирования


class SubmitModelWithMemory(ModelWithMemory):
    """
    Класс для работы с долгосрочной памятью в диалоговой системе
    """

    def __init__(self, model_path: str) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        self.rag_interface = RAGInterface()

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Фильтрует и сохраняет личную информацию из сообщений
        Группирует сообщения по сессиям и склеивает их
        """
        # Группируем сообщения по сессиям
        session_messages = {}
        session_has_personal = {}  # Отслеживаем, есть ли в сессии личные сообщения
        
        for msg in messages:
            if msg.role == "user":
                # Проверяем кэш для ускорения
                cached_result = self.storage.check_cache(msg.content)
                
                if cached_result is None:
                    filter_result = is_personal_message(msg.content)
                    self.storage.add_to_cache(msg.content, filter_result)
                    cached_result = filter_result
                
                if cached_result:
                    # Используем session_id из сообщения, если он есть
                    session_id = msg.session_id if msg.session_id else self.storage.increment_session(dialogue_id)
                    
                    # Регистрируем сессию, если используем session_id из сообщения
                    if msg.session_id:
                        self.storage.register_session(dialogue_id, session_id)
                    
                    # Группируем по session_id
                    if session_id not in session_messages:
                        session_messages[session_id] = []
                        session_has_personal[session_id] = False
                    
                    # Если сообщение прошло фильтрацию, значит содержит личную информацию
                    session_has_personal[session_id] = True
                    session_messages[session_id].append(msg.content)
        
        # Склеиваем сообщения по сессиям
        processed_messages = []
        for session_id, contents in session_messages.items():
            # Склеиваем все сообщения сессии
            combined_content = self._combine_session_messages(contents)
            
            # Маркер [ЛИЧНОЕ] временно отключен
            # if session_has_personal.get(session_id, False):
            #     personal_marker = get_personal_info_marker()
            #     combined_content = f"{personal_marker} {combined_content}"
            
            processed_msg = Message(
                role="user",
                content=f"{get_session_marker_prompt(int(session_id))} {combined_content}",
                session_id=session_id
            )
            processed_messages.append(processed_msg)
        
        self.storage.add_to_memory(dialogue_id, processed_messages)

    def _combine_session_messages(self, contents: List[str]) -> str:
        """
        Склеивает сообщения из одной сессии, добавляя точки где нужно
        
        Args:
            contents: Список содержимого сообщений
            
        Returns:
            Склеенное содержимое
        """
        if not contents:
            return ""
        
        # Знаки препинания для проверки
        punctuation_marks = {'.', '!', '?', ':', ';'}
        
        combined_parts = []
        for content in contents:
            content = content.strip()
            if content:
                # Проверяем последний символ
                if content and content[-1] not in punctuation_marks:
                    content += "."
                combined_parts.append(content)
        
        return " ".join(combined_parts)

    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память диалога и кэш при необходимости
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
        rag_prompt = self.rag_interface.answer_question(question, dialogue_id, memory)
        
        # Создаем контекст для модели
        context_with_memory = [Message('system', rag_prompt)]
        
        # Генерируем ответ через модель
        answer = self.model_inference.inference(context_with_memory)
        return answer

    def answer_to_question_mock(self, dialogue_id: str, question: str) -> str:
        """
        Возвращает промпт который отправляется в модель (не ответ!)
        Использует RAG систему для генерации умного промпта
        """
        # Получаем все сообщения из памяти
        memory = self.storage.get_memory(dialogue_id)
        
        if not memory:
            return "У меня нет информации для ответа на этот вопрос."
        
        # Используем RAG систему для генерации промпта
        rag_prompt = self.rag_interface.answer_question(question, dialogue_id, memory)
        return rag_prompt
    
    def get_rag_stats(self, dialogue_id: str) -> dict:
        """
        Получает статистику RAG системы для диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь со статистикой
        """
        memory = self.storage.get_memory(dialogue_id)
        if not memory:
            return {'error': 'Нет данных в памяти'}
        
        # Получаем статистику RAG системы
        rag_stats = self.rag_interface.get_system_stats()
        
        # Добавляем статистику по диалогу
        dialogue_stats = {
            'dialogue_id': dialogue_id,
            'memory_messages': len(memory),
            'sessions_count': self.rag_interface.get_session_count(dialogue_id),
            'rag_config': rag_stats.get('config', {}),
            'available_topics': rag_stats.get('available_topics', [])
        }
        
        return dialogue_stats
    
    def analyze_question(self, dialogue_id: str, question: str) -> dict:
        """
        Анализирует вопрос с помощью RAG системы
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            
        Returns:
            Словарь с анализом вопроса
        """
        memory = self.storage.get_memory(dialogue_id)
        if not memory:
            return {'error': 'Нет данных в памяти'}
        
        # Получаем анализ вопроса
        analysis = self.rag_interface.get_question_context(question, dialogue_id, memory)
        return analysis