# src/submit/modules/storage/module.py

from ...core.interfaces import IStorage, ProcessingResult
from ...core.container import container
from typing import List
from models import Message

class StorageModule(IStorage):
    """Модуль хранилища"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._init_storage()
    
    def _init_storage(self):
        """Инициализирует хранилище"""
        self.memory = {}
        self.sessions = {}
        self.cache = {}
    
    def add_messages(self, dialogue_id: str, messages: List[Message]) -> ProcessingResult:
        try:
            # Фильтрация
            filtered = self._filter_messages(messages)
            
            # Группировка по сессиям
            sessions = self._group_by_sessions(filtered, dialogue_id)
            
            # Сохранение
            if dialogue_id not in self.memory:
                self.memory[dialogue_id] = []
            
            self.memory[dialogue_id].extend(filtered)
            self.sessions[dialogue_id] = sessions
            
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
    
    def get_messages(self, dialogue_id: str) -> ProcessingResult:
        try:
            messages = self.memory.get(dialogue_id, [])
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
    
    def clear(self, dialogue_id: str) -> ProcessingResult:
        try:
            if dialogue_id in self.memory:
                del self.memory[dialogue_id]
            if dialogue_id in self.sessions:
                del self.sessions[dialogue_id]
            
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
    
    def _filter_messages(self, messages: List[Message]) -> List[Message]:
        """Фильтрует сообщения"""
        filtered = []
        for msg in messages:
            if msg.role == "user" and self._is_personal(msg.content):
                filtered.append(msg)
        return filtered
    
    def _is_personal(self, content: str) -> bool:
        """Проверяет, содержит ли текст личную информацию"""
        # Упрощенная логика для примера
        personal_markers = ['я', 'меня', 'мой', 'моя', 'мне']
        content_lower = content.lower()
        return any(marker in content_lower for marker in personal_markers)
    
    def _group_by_sessions(self, messages: List[Message], dialogue_id: str) -> dict:
        """Группирует сообщения по сессиям"""
        sessions = {}
        for msg in messages:
            session_id = getattr(msg, 'session_id', '0')
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(msg)
        return sessions