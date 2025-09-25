"""
Модели данных для управления сессиями
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class SessionInfo:
    """Информация о сессии"""
    session_id: str
    dialogue_id: str
    message_count: int
    created_at: Optional[str] = None
    last_activity: Optional[str] = None


@dataclass
class SessionStats:
    """Статистика по сессиям диалога"""
    dialogue_id: str
    total_sessions: int
    total_messages: int
    avg_messages_per_session: float
    session_ids: List[str]


class SessionRegistry:
    """Реестр сессий для диалога"""
    
    def __init__(self):
        # Счетчики сессий для каждого диалога
        self.session_counters = defaultdict(int)
        
        # Реестр зарегистрированных сессий
        self.registered_sessions = defaultdict(set)
        
        # Детальная информация о сессиях
        self.session_info = defaultdict(dict)  # {dialogue_id: {session_id: SessionInfo}}
    
    def increment_session(self, dialogue_id: str) -> int:
        """Увеличивает счетчик сессий и возвращает новый номер"""
        self.session_counters[dialogue_id] += 1
        return self.session_counters[dialogue_id]
    
    def register_session(self, dialogue_id: str, session_id: str) -> None:
        """Регистрирует сессию с заданным ID"""
        if session_id not in self.registered_sessions[dialogue_id]:
            self.registered_sessions[dialogue_id].add(session_id)
            
            # Обновляем счетчик, если session_id больше текущего
            try:
                session_num = int(session_id)
                self.session_counters[dialogue_id] = max(
                    self.session_counters[dialogue_id], 
                    session_num
                )
            except ValueError:
                # Если session_id не число, просто увеличиваем счетчик
                self.session_counters[dialogue_id] += 1
    
    def get_current_session(self, dialogue_id: str) -> int:
        """Возвращает текущий номер сессии"""
        return self.session_counters[dialogue_id]
    
    def get_session_count(self, dialogue_id: str) -> int:
        """Возвращает количество сессий в диалоге"""
        return len(self.registered_sessions[dialogue_id])
    
    def get_session_ids(self, dialogue_id: str) -> List[str]:
        """Возвращает список ID сессий диалога"""
        return list(self.registered_sessions[dialogue_id])
    
    def add_session_info(self, dialogue_id: str, session_id: str, 
                        message_count: int = 0) -> None:
        """Добавляет информацию о сессии"""
        session_info = SessionInfo(
            session_id=session_id,
            dialogue_id=dialogue_id,
            message_count=message_count
        )
        self.session_info[dialogue_id][session_id] = session_info
    
    def get_session_info(self, dialogue_id: str, session_id: str) -> Optional[SessionInfo]:
        """Получает информацию о конкретной сессии"""
        return self.session_info[dialogue_id].get(session_id)
    
    def get_session_stats(self, dialogue_id: str) -> SessionStats:
        """Получает статистику по сессиям диалога"""
        session_ids = self.get_session_ids(dialogue_id)
        total_sessions = len(session_ids)
        
        # Подсчитываем общее количество сообщений
        total_messages = sum(
            info.message_count 
            for info in self.session_info[dialogue_id].values()
        )
        
        avg_messages = total_messages / total_sessions if total_sessions > 0 else 0
        
        return SessionStats(
            dialogue_id=dialogue_id,
            total_sessions=total_sessions,
            total_messages=total_messages,
            avg_messages_per_session=avg_messages,
            session_ids=session_ids
        )
    
    def clear_dialogue_sessions(self, dialogue_id: str) -> None:
        """Очищает все сессии диалога"""
        self.session_counters[dialogue_id] = 0
        self.registered_sessions[dialogue_id].clear()
        self.session_info[dialogue_id].clear()
    
    def is_session_registered(self, dialogue_id: str, session_id: str) -> bool:
        """Проверяет, зарегистрирована ли сессия"""
        return session_id in self.registered_sessions[dialogue_id]
