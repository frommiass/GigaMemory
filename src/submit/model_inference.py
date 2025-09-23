from collections import defaultdict
from dataclasses import asdict
from typing import List

from models import Message
from submit_interface import ModelWithMemory


class SubmitModelWithMemory(ModelWithMemory):

    def __init__(self, model_path: str) -> None:
        self.basic_memory = defaultdict(list)
        self.model_path = model_path
        print(f"⚠️  ЗАГЛУШКА: Модель {model_path} не загружена, все ответы будут 'У меня нет такой информации'")

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """Сохраняет сообщения в память (заглушка)"""
        self.basic_memory[dialogue_id] += messages
        print(f"📝 ЗАГЛУШКА: Сохранено {len(messages)} сообщений для диалога {dialogue_id}")

    def extract(self, dialogue_id: str) -> List[Message]:
        """Извлекает историю диалога (заглушка)"""
        memory = self.basic_memory.get(dialogue_id, [])
        memory = [asdict(msg) for msg in memory]
        memory = "\n".join([f"{msg['role']}: {msg['content']}" for msg in memory])
        system_memory_prompt = "Твоя задача - ответить на вопрос пользователя. Для этого тебе подается на вход твоя история общения с пользователем." \
                               "Пользователь разрешил использовать ее для ответа на вопрос. Используй историю диалога, чтобы ответить на вопрос.\n" \
                               f"История диалога: \n{memory}"

        context_with_memory = [Message('system', system_memory_prompt)]

        return context_with_memory

    def clear_memory(self, dialogue_id: str) -> None:
        """Очищает память диалога (заглушка)"""
        self.basic_memory[dialogue_id] = []
        print(f"🗑️  ЗАГЛУШКА: Память диалога {dialogue_id} очищена")

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос (заглушка - всегда возвращает 'У меня нет такой информации')"""
        print(f"❓ ЗАГЛУШКА: Получен вопрос '{question}' для диалога {dialogue_id}")
        return "У меня нет такой информации"

    def _inference(self, messages: List[Message]) -> str:
        """Инференс модели (заглушка)"""
        print(f"🤖 ЗАГЛУШКА: Инференс с {len(messages)} сообщениями")
        return "У меня нет такой информации"