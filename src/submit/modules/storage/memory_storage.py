# src/submit/storage/memory_storage.py
"""
Модуль для управления памятью и кэшем в GigaMemory
Упрощенная версия без дублирования SessionGrouper
"""
from typing import List, Dict, Any
from models import Message
from .memory_models import MemoryRegistry


class MemoryStorage:
    """Класс для управления памятью и кэшированием"""
    
    def __init__(self):
        # Реестр памяти
        self.memory_registry = MemoryRegistry()
    
    def add_to_memory(self, dialogue_id: str, messages: List[Message]) -> None:
        """Добавляет сообщения в память диалога"""
        self.memory_registry.add_to_memory(dialogue_id, messages)
    
    def get_memory(self, dialogue_id: str) -> List[Message]:
        """Получает все сообщения из памяти диалога"""
        return self.memory_registry.get_memory(dialogue_id)
    
    def check_cache(self, content: str) -> Any:
        """
        Проверяет наличие результата в кэше
        
        Args:
            content: Содержимое для проверки
            
        Returns:
            Результат из кэша или None если не найден
        """
        return self.memory_registry.check_cache(content)
    
    def add_to_cache(self, content: str, result: Any) -> None:
        """
        Добавляет результат в кэш
        
        Args:
            content: Содержимое (ключ)
            result: Результат для кэширования
        """
        self.memory_registry.add_to_cache(content, result)
    
    def cleanup_cache(self) -> None:
        """Очищает половину кэша при переполнении"""
        self.memory_registry.cleanup_cache()
    
    def clear_dialogue_memory(self, dialogue_id: str) -> None:
        """Полностью очищает память конкретного диалога"""
        self.memory_registry.clear_dialogue_memory(dialogue_id)
    
    def clear_all_cache(self) -> None:
        """Полностью очищает весь кэш"""
        self.memory_registry.clear_all_cache()
    
    def get_cache_size(self) -> int:
        """Возвращает текущий размер кэша"""
        return self.memory_registry.get_cache_size()
    
    def get_memory_stats(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Возвращает статистику по памяти диалога
        
        Returns:
            Словарь со статистикой
        """
        memory_stats = self.memory_registry.get_memory_stats(dialogue_id)
        
        return {
            'messages_count': memory_stats.messages_count,
            'sessions_count': memory_stats.sessions_count,
            'cache_size': memory_stats.cache_size,
            'personal_messages': memory_stats.personal_messages,
            'non_personal_messages': memory_stats.non_personal_messages
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по кэшу
        
        Returns:
            Словарь со статистикой кэша
        """
        cache_stats = self.memory_registry.get_cache_stats()
        
        return {
            'total_entries': cache_stats.total_entries,
            'hit_rate': cache_stats.hit_rate,
            'memory_usage': cache_stats.memory_usage,
            'oldest_entry': cache_stats.oldest_entry,
            'newest_entry': cache_stats.newest_entry
        }
    
    def get_session_stats(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Возвращает статистику по сессиям диалога
        Делегирует работу SessionGrouper через StorageModule
        
        Returns:
            Словарь со статистикой сессий
        """
        # Этот метод будет вызываться из StorageModule
        # который имеет доступ к SessionGrouper
        return {
            'dialogue_id': dialogue_id,
            'total_sessions': 0,  # Будет заполнено из SessionGrouper
            'total_messages': len(self.get_memory(dialogue_id)),
            'avg_messages_per_session': 0.0,
            'session_ids': []
        }