# modules/storage/module.py

from typing import List, Dict, Any
from models import Message
from core.interfaces import IStorage, ProcessingResult

from .memory_storage import MemoryStorage
from .session_manager import SessionManager
from .message_filter import MessageFilter
from .filters.session_grouper import SessionGrouper


class StorageModule(IStorage):
    """Модуль управления хранилищем и сессиями"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = MemoryStorage()
        self.sessions = SessionManager()
        self.filter = MessageFilter()
        self.grouper = SessionGrouper()
    
    def store_messages(self, dialogue_id: str, messages: List[Message]) -> ProcessingResult:
        """Сохраняет сообщения с фильтрацией"""
        try:
            # Фильтруем сообщения
            filtered = self.filter.filter_messages(messages)
            
            # Группируем по сессиям
            sessions = self.grouper.group_messages_by_sessions(filtered, dialogue_id)
            
            # Сохраняем
            self.memory.add_to_memory(dialogue_id, filtered)
            
            return ProcessingResult(
                success=True,
                data=filtered,
                metadata={
                    'total': len(messages),
                    'filtered': len(filtered),
                    'sessions': len(sessions)
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_dialogue_messages(self, dialogue_id: str) -> ProcessingResult:
        """Получает все сообщения диалога"""
        try:
            messages = self.memory.get_memory(dialogue_id)
            return ProcessingResult(
                success=True,
                data=messages,
                metadata={'count': len(messages)}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )
    
    def get_dialogue_sessions(self, dialogue_id: str) -> ProcessingResult:
        """Получает сессии диалога"""
        try:
            messages = self.memory.get_memory(dialogue_id)
            sessions = self.grouper.group_messages_by_sessions(messages, dialogue_id)
            return ProcessingResult(
                success=True,
                data=sessions,
                metadata={'sessions_count': len(sessions)}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def clear_dialogue(self, dialogue_id: str) -> ProcessingResult:
        """Очищает данные диалога"""
        try:
            self.memory.clear_dialogue_memory(dialogue_id)
            self.sessions.clear_dialogue_sessions(dialogue_id)
            return ProcessingResult(
                success=True,
                data=None,
                metadata={'cleared': True}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
