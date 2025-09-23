from typing import List
from models import Message
from submit_interface import ModelWithMemory

from .smart_memory import SmartMemory, SmartMemoryConfig


class SubmitModelWithMemory(ModelWithMemory):
    """
    Интеллектуальная система памяти с векторным поиском, извлечением фактов и сжатием
    """

    def __init__(self, model_path: str) -> None:
        # Конфигурация умной памяти
        config = SmartMemoryConfig()
        config.use_vector_search = True
        config.use_fact_extraction = True
        config.use_compression = True
        
        # Создаем умную память
        self.smart_memory = SmartMemory(model_path, config)
        
        # Временное хранилище для батчевой записи
        self.pending_messages = {}
    
    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """Записывает сообщения в интеллектуальную память"""
        
        # Накапливаем сообщения для батчевой обработки
        if dialogue_id not in self.pending_messages:
            self.pending_messages[dialogue_id] = []
        
        self.pending_messages[dialogue_id].extend(messages)
    
    def clear_memory(self, dialogue_id: str) -> None:
        """Очищает память диалога"""
        
        # Обрабатываем накопленные сообщения перед очисткой
        if dialogue_id in self.pending_messages:
            messages = self.pending_messages[dialogue_id]
            if messages:
                # Обрабатываем диалог полным циклом
                stats = self.smart_memory.process_dialogue(dialogue_id, messages)
                print(f"📊 Обработано: {stats['sessions_count']} сессий, "
                      f"{stats['facts_extracted']} фактов, "
                      f"сжатие {stats['compression_ratio']:.2f}")
            
            del self.pending_messages[dialogue_id]
    
    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос используя интеллектуальную систему"""
        
        # Сначала обрабатываем накопленные сообщения
        if dialogue_id in self.pending_messages:
            messages = self.pending_messages[dialogue_id]
            if messages:
                self.smart_memory.process_dialogue(dialogue_id, messages)
        
        # Отвечаем на вопрос
        return self.smart_memory.answer_question(dialogue_id, question)