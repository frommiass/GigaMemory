"""
Менеджер для единого управления сессиями
"""
from typing import List, Dict, Optional
from .session_models import SessionRegistry, SessionInfo, SessionStats


class SessionManager:
    """Единый менеджер для управления сессиями"""
    
    def __init__(self):
        self.registry = SessionRegistry()
    
    def increment_session(self, dialogue_id: str) -> int:
        """Увеличивает счетчик сессий и возвращает новый номер"""
        return self.registry.increment_session(dialogue_id)
    
    def register_session(self, dialogue_id: str, session_id: str) -> None:
        """Регистрирует сессию с заданным ID"""
        self.registry.register_session(dialogue_id, session_id)
    
    def get_current_session(self, dialogue_id: str) -> int:
        """Возвращает текущий номер сессии"""
        return self.registry.get_current_session(dialogue_id)
    
    def get_session_count(self, dialogue_id: str) -> int:
        """Возвращает количество сессий в диалоге"""
        return self.registry.get_session_count(dialogue_id)
    
    def get_session_ids(self, dialogue_id: str) -> List[str]:
        """Возвращает список ID сессий диалога"""
        return self.registry.get_session_ids(dialogue_id)
    
    def add_session_info(self, dialogue_id: str, session_id: str, 
                        message_count: int = 0) -> None:
        """Добавляет информацию о сессии"""
        self.registry.add_session_info(dialogue_id, session_id, message_count)
    
    def get_session_info(self, dialogue_id: str, session_id: str) -> Optional[SessionInfo]:
        """Получает информацию о конкретной сессии"""
        return self.registry.get_session_info(dialogue_id, session_id)
    
    def get_session_stats(self, dialogue_id: str) -> SessionStats:
        """Получает статистику по сессиям диалога"""
        return self.registry.get_session_stats(dialogue_id)
    
    def clear_dialogue_sessions(self, dialogue_id: str) -> None:
        """Очищает все сессии диалога"""
        self.registry.clear_dialogue_sessions(dialogue_id)
    
    def is_session_registered(self, dialogue_id: str, session_id: str) -> bool:
        """Проверяет, зарегистрирована ли сессия"""
        return self.registry.is_session_registered(dialogue_id, session_id)
    
    def get_all_dialogues(self) -> List[str]:
        """Возвращает список всех диалогов"""
        return list(self.registry.session_counters.keys())
    
    def get_global_stats(self) -> Dict[str, int]:
        """Возвращает глобальную статистику по всем диалогам"""
        total_dialogues = len(self.registry.session_counters)
        total_sessions = sum(self.registry.session_counters.values())
        
        return {
            'total_dialogues': total_dialogues,
            'total_sessions': total_sessions,
            'avg_sessions_per_dialogue': total_sessions / total_dialogues if total_dialogues > 0 else 0
        }
