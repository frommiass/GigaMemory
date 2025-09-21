"""
Модуль для управления кэшем и хранением памяти в GigaMemory
"""
from collections import defaultdict
from typing import List, Dict, Any
from models import Message


class MemoryStorage:
    """Класс для управления памятью и кэшированием"""
    
    def __init__(self):
        # Основное хранилище памяти по диалогам
        self.basic_memory = defaultdict(list)
        
        # Счетчики сессий для каждого диалога
        self.session_counters = defaultdict(int)
        
        # Кэш результатов фильтрации
        self.filter_cache = {}
        
        # Настройки кэша
        self.cache_max_size = 10000
        self.cache_cleanup_ratio = 0.5
    
    def add_to_memory(self, dialogue_id: str, messages: List[Message]) -> None:
        """Добавляет сообщения в память диалога"""
        self.basic_memory[dialogue_id] += messages
    
    def get_memory(self, dialogue_id: str) -> List[Message]:
        """Получает все сообщения из памяти диалога"""
        return self.basic_memory.get(dialogue_id, [])
    
    def increment_session(self, dialogue_id: str) -> int:
        """Увеличивает счетчик сессий и возвращает новый номер"""
        self.session_counters[dialogue_id] += 1
        return self.session_counters[dialogue_id]
    
    def get_current_session(self, dialogue_id: str) -> int:
        """Возвращает текущий номер сессии"""
        return self.session_counters[dialogue_id]
    
    def check_cache(self, content: str) -> Any:
        """
        Проверяет наличие результата в кэше
        
        Args:
            content: Содержимое для проверки
            
        Returns:
            Результат из кэша или None если не найден
        """
        cache_key = hash(content)
        return self.filter_cache.get(cache_key)
    
    def add_to_cache(self, content: str, result: Any) -> None:
        """
        Добавляет результат в кэш
        
        Args:
            content: Содержимое (ключ)
            result: Результат для кэширования
        """
        cache_key = hash(content)
        self.filter_cache[cache_key] = result
        
        # Очистка кэша при переполнении
        if len(self.filter_cache) > self.cache_max_size:
            self.cleanup_cache()
    
    def cleanup_cache(self) -> None:
        """Очищает половину кэша при переполнении"""
        items = list(self.filter_cache.items())
        keep_size = int(len(items) * self.cache_cleanup_ratio)
        # Оставляем вторую половину (более новые записи)
        self.filter_cache = dict(items[keep_size:])
    
    def clear_dialogue_memory(self, dialogue_id: str) -> None:
        """Полностью очищает память конкретного диалога"""
        self.basic_memory[dialogue_id] = []
        self.session_counters[dialogue_id] = 0
    
    def clear_all_cache(self) -> None:
        """Полностью очищает весь кэш"""
        self.filter_cache.clear()
    
    def get_cache_size(self) -> int:
        """Возвращает текущий размер кэша"""
        return len(self.filter_cache)
    
    def get_memory_stats(self, dialogue_id: str) -> Dict[str, int]:
        """
        Возвращает статистику по памяти диалога
        
        Returns:
            Словарь со статистикой
        """
        return {
            'messages_count': len(self.basic_memory.get(dialogue_id, [])),
            'sessions_count': self.session_counters[dialogue_id],
            'cache_size': len(self.filter_cache)
        }
