"""
Модели данных для управления памятью и кэшем
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from collections import defaultdict


@dataclass
class MemoryEntry:
    """Запись в памяти"""
    dialogue_id: str
    message_id: str
    content: str
    role: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    is_personal: bool = True


@dataclass
class CacheEntry:
    """Запись в кэше"""
    content: str
    result: Any
    created_at: Optional[str] = None
    access_count: int = 0


@dataclass
class MemoryStats:
    """Статистика по памяти диалога"""
    dialogue_id: str
    messages_count: int
    sessions_count: int
    cache_size: int
    personal_messages: int
    non_personal_messages: int


@dataclass
class CacheStats:
    """Статистика по кэшу"""
    total_entries: int
    hit_rate: float
    memory_usage: int
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None


class MemoryRegistry:
    """Реестр памяти для диалогов"""
    
    def __init__(self):
        # Основное хранилище памяти по диалогам
        self.basic_memory = defaultdict(list)
        
        # Кэш результатов фильтрации
        self.filter_cache = {}
        
        # Настройки кэша
        self.cache_max_size = 10000
        self.cache_cleanup_ratio = 0.5
        
        # Статистика доступа к кэшу
        self.cache_hits = 0
        self.cache_misses = 0
    
    def add_to_memory(self, dialogue_id: str, messages: List[Any]) -> None:
        """Добавляет сообщения в память диалога"""
        self.basic_memory[dialogue_id] += messages
    
    def get_memory(self, dialogue_id: str) -> List[Any]:
        """Получает все сообщения из памяти диалога"""
        return self.basic_memory.get(dialogue_id, [])
    
    def clear_dialogue_memory(self, dialogue_id: str) -> None:
        """Полностью очищает память конкретного диалога"""
        self.basic_memory[dialogue_id] = []
    
    def check_cache(self, content: str) -> Any:
        """Проверяет наличие результата в кэше"""
        cache_key = hash(content)
        if cache_key in self.filter_cache:
            self.cache_hits += 1
            return self.filter_cache[cache_key]
        else:
            self.cache_misses += 1
            return None
    
    def add_to_cache(self, content: str, result: Any) -> None:
        """Добавляет результат в кэш"""
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
    
    def clear_all_cache(self) -> None:
        """Полностью очищает весь кэш"""
        self.filter_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_size(self) -> int:
        """Возвращает текущий размер кэша"""
        return len(self.filter_cache)
    
    def get_memory_stats(self, dialogue_id: str) -> MemoryStats:
        """Возвращает статистику по памяти диалога"""
        messages = self.basic_memory.get(dialogue_id, [])
        
        # Подсчитываем личные и неличные сообщения
        personal_count = 0
        non_personal_count = 0
        
        for msg in messages:
            if hasattr(msg, 'is_personal'):
                if msg.is_personal:
                    personal_count += 1
                else:
                    non_personal_count += 1
            else:
                personal_count += 1  # По умолчанию считаем личными
        
        return MemoryStats(
            dialogue_id=dialogue_id,
            messages_count=len(messages),
            sessions_count=0,  # Будет заполнено из SessionRegistry
            cache_size=len(self.filter_cache),
            personal_messages=personal_count,
            non_personal_messages=non_personal_count
        )
    
    def get_cache_stats(self) -> CacheStats:
        """Возвращает статистику по кэшу"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            total_entries=len(self.filter_cache),
            hit_rate=hit_rate,
            memory_usage=len(str(self.filter_cache)),
            oldest_entry=None,  # Можно добавить timestamp для отслеживания
            newest_entry=None
        )
