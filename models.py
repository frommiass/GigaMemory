"""
Модели данных для системы GigaMemory
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Message:
    """Сообщение в диалоге"""
    role: str  # "user" или "assistant"
    content: str
    session_id: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Session:
    """Сессия диалога"""
    id: str
    messages: List[Message]
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Dialogue:
    """Диалог пользователя"""
    id: str
    sessions: List[Session]
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Fact:
    """Извлеченный факт"""
    id: str
    content: str
    dialogue_id: str
    session_id: str
    confidence: float
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Dialog:
    """Диалог (альтернативное название для Dialogue)"""
    id: str
    sessions: List[Session]
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SearchResult:
    """Результат поиска"""
    id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
