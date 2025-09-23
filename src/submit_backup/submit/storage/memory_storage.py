"""
Модуль для управления памятью и кэшем в GigaMemory
Использует SessionManager для управления сессиями
"""
from typing import List, Dict, Any
from models import Message

from .session_manager import SessionManager
from .memory_models import MemoryRegistry, MemoryStats, CacheStats


class MemoryStorage:
    """Класс для управления памятью и кэшированием"""
    
    def __init__(self):
        # Менеджер сессий
        self.session_manager = SessionManager()
        
        # Реестр памяти
        self.memory_registry = MemoryRegistry()
    
    def add_to_memory(self, dialogue_id: str, messages: List[Message]) -> None:
        """Добавляет сообщения в память диалога"""
        self.memory_registry.add_to_memory(dialogue_id, messages)
    
    def get_memory(self, dialogue_id: str) -> List[Message]:
        """Получает все сообщения из памяти диалога"""
        return self.memory_registry.get_memory(dialogue_id)
    
    def increment_session(self, dialogue_id: str) -> int:
        """Увеличивает счетчик сессий и возвращает новый номер (для обратной совместимости)"""
        return self.session_manager.increment_session(dialogue_id)
    
    def register_session(self, dialogue_id: str, session_id: str) -> None:
        """Регистрирует сессию с заданным ID"""
        self.session_manager.register_session(dialogue_id, session_id)
    
    def get_current_session(self, dialogue_id: str) -> int:
        """Возвращает текущий номер сессии"""
        return self.session_manager.get_current_session(dialogue_id)
    
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
        self.session_manager.clear_dialogue_sessions(dialogue_id)
    
    def clear_all_cache(self) -> None:
        """Полностью очищает весь кэш"""
        self.memory_registry.clear_all_cache()
    
    def get_cache_size(self) -> int:
        """Возвращает текущий размер кэша"""
        return self.memory_registry.get_cache_size()
    
    def get_memory_stats(self, dialogue_id: str) -> Dict[str, int]:
        """
        Возвращает статистику по памяти диалога
        
        Returns:
            Словарь со статистикой
        """
        memory_stats = self.memory_registry.get_memory_stats(dialogue_id)
        session_stats = self.session_manager.get_session_stats(dialogue_id)
        
        return {
            'messages_count': memory_stats.messages_count,
            'sessions_count': session_stats.total_sessions,
            'cache_size': memory_stats.cache_size,
            'personal_messages': memory_stats.personal_messages,
            'non_personal_messages': memory_stats.non_personal_messages
        }
    
    def get_session_stats(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Возвращает статистику по сессиям диалога
        
        Returns:
            Словарь со статистикой сессий
        """
        session_stats = self.session_manager.get_session_stats(dialogue_id)
        
        return {
            'dialogue_id': session_stats.dialogue_id,
            'total_sessions': session_stats.total_sessions,
            'total_messages': session_stats.total_messages,
            'avg_messages_per_session': session_stats.avg_messages_per_session,
            'session_ids': session_stats.session_ids
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
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Возвращает глобальную статистику по всем диалогам
        
        Returns:
            Словарь с глобальной статистикой
        """
        session_global = self.session_manager.get_global_stats()
        cache_stats = self.memory_registry.get_cache_stats()
        
        return {
            'total_dialogues': session_global['total_dialogues'],
            'total_sessions': session_global['total_sessions'],
            'avg_sessions_per_dialogue': session_global['avg_sessions_per_dialogue'],
            'cache_entries': cache_stats.total_entries,
            'cache_hit_rate': cache_stats.hit_rate
        }
