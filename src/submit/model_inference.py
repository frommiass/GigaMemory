from dataclasses import asdict
from typing import List

from models import Message
from submit_interface import ModelWithMemory

from .memory_storage import MemoryStorage
from .message_filter import filter_user_message, _check_markers
from .regex_patterns import PERSONAL_MARKERS
from .prompts import (
    get_session_marker_prompt,
    get_personal_info_marker
)
from .prompts.fallback_prompts import get_fallback_prompt
from .llm_inference import ModelInference


class SubmitModelWithMemory(ModelWithMemory):
    """
    Класс для работы с долгосрочной памятью в диалоговой системе
    """

    def __init__(self, model_path: str) -> None:
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)

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
                    filter_result = filter_user_message(msg.content)
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
                    
                    # Проверяем, содержит ли сообщение личные местоимения
                    if _check_markers(msg.content.lower(), PERSONAL_MARKERS):
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
        Генерирует ответ на вопрос используя накопленную память
        """
        memory = self.storage.get_memory(dialogue_id)
        memory = [asdict(msg) for msg in memory]
        memory_text = "\n".join([msg['content'] for msg in memory])
        
        system_memory_prompt = get_fallback_prompt(question, memory_text)
        context_with_memory = [Message('system', system_memory_prompt)]
        
        answer = self.model_inference.inference(context_with_memory)
        return answer

    def answer_to_question_mock(self, dialogue_id: str, question: str) -> str:
        """
        Возвращает промпт который отправляется в модель (не ответ!)
        """
        memory = self.storage.get_memory(dialogue_id)
        memory = [asdict(msg) for msg in memory]
        memory_text = "\n".join([msg['content'] for msg in memory])
        
        system_memory_prompt = get_fallback_prompt(question, memory_text)
        return system_memory_prompt