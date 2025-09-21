from dataclasses import asdict
from typing import List

from models import Message
from submit_interface import ModelWithMemory

from .memory_storage import MemoryStorage
from .message_filter import filter_user_message
from .prompts import (
    get_memory_extraction_prompt,
    get_session_marker_prompt
)
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
        """
        current_session = self.storage.increment_session(dialogue_id)
        
        processed_messages = []
        for msg in messages:
            if msg.role == "user":
                # Проверяем кэш для ускорения
                cached_result = self.storage.check_cache(msg.content)
                
                if cached_result is None:
                    filter_result = filter_user_message(msg.content)
                    self.storage.add_to_cache(msg.content, filter_result)
                    cached_result = filter_result
                
                if cached_result:
                    processed_msg = Message(
                        role=msg.role,
                        content=f"{get_session_marker_prompt(current_session)} {msg.content}"
                    )
                    processed_messages.append(processed_msg)
        
        self.storage.add_to_memory(dialogue_id, processed_messages)

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
        
        system_memory_prompt = get_memory_extraction_prompt(question, memory_text)
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
        
        system_memory_prompt = get_memory_extraction_prompt(question, memory_text)
        return system_memory_prompt