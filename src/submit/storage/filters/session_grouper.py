"""
Модуль для группировки сообщений по сессиям
Основан на существующей логике из memory_storage.py и model_inference.py
"""
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from models import Message

from ..session_manager import SessionManager
from .message_cleaner import is_copy_paste_content


class SessionGrouper:
    """Класс для группировки сообщений по сессиям"""
    
    def __init__(self):
        # Менеджер сессий
        self.session_manager = SessionManager()
        # Группированные сообщения по диалогам и сессиям
        self.grouped_sessions = defaultdict(lambda: defaultdict(list))
    
    def group_messages_by_sessions(self, messages: List[Message], dialogue_id: str) -> Dict[str, List[Message]]:
        """
        Группирует сообщения по сессиям
        
        Args:
            messages: Список сообщений для группировки
            dialogue_id: ID диалога
            
        Returns:
            Словарь {session_id: [messages]}
        """
        sessions = defaultdict(list)
        
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                # Используем session_id из сообщения, если он есть
                if msg.session_id:
                    session_id = msg.session_id
                    # Регистрируем сессию
                    self.session_manager.register_session(dialogue_id, session_id)
                else:
                    # Создаем новую сессию
                    session_id = str(self.session_manager.increment_session(dialogue_id))
                
                # Сохраняем сообщение без маркера (маркер добавим позже)
                marked_msg = Message(
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id
                )
                
                sessions[session_id].append(marked_msg)
        
        # Сохраняем в группированном виде
        self.grouped_sessions[dialogue_id] = sessions
        
        return dict(sessions)
    
    def get_session_messages(self, dialogue_id: str, session_id: str) -> List[Message]:
        """
        Получает сообщения конкретной сессии
        
        Args:
            dialogue_id: ID диалога
            session_id: ID сессии
            
        Returns:
            Список сообщений сессии
        """
        return self.grouped_sessions.get(dialogue_id, {}).get(session_id, [])
    
    def get_all_sessions(self, dialogue_id: str) -> Dict[str, List[Message]]:
        """
        Получает все сессии диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь всех сессий диалога
        """
        return dict(self.grouped_sessions.get(dialogue_id, {}))
    
    def get_session_count(self, dialogue_id: str) -> int:
        """
        Получает количество сессий в диалоге
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Количество сессий
        """
        return self.session_manager.get_session_count(dialogue_id)
    
    def get_session_ids(self, dialogue_id: str) -> List[str]:
        """
        Получает список ID сессий диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Список ID сессий
        """
        return self.session_manager.get_session_ids(dialogue_id)
    
    def clear_dialogue_sessions(self, dialogue_id: str) -> None:
        """Очищает все сессии диалога"""
        self.grouped_sessions[dialogue_id] = defaultdict(list)
        self.session_manager.clear_dialogue_sessions(dialogue_id)
    
    def get_session_stats(self, dialogue_id: str) -> Dict[str, int]:
        """
        Получает статистику по сессиям диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь со статистикой
        """
        sessions = self.grouped_sessions.get(dialogue_id, {})
        total_messages = sum(len(msgs) for msgs in sessions.values())
        
        return {
            'sessions_count': len(sessions),
            'total_messages': total_messages,
            'avg_messages_per_session': total_messages / len(sessions) if sessions else 0
        }


def group_messages_by_sessions_simple(messages: List[Message], dialogue_id: str) -> Dict[str, List[Message]]:
    """
    Простая функция для группировки сообщений по сессиям
    (для обратной совместимости)
    
    Args:
        messages: Список сообщений
        dialogue_id: ID диалога
        
    Returns:
        Словарь {session_id: [messages]}
    """
    grouper = SessionGrouper()
    return grouper.group_messages_by_sessions(messages, dialogue_id)


def extract_session_content(session_messages: List[Message]) -> str:
    """
    Извлекает текстовое содержимое сессии (только сообщения пользователя)
    
    Args:
        session_messages: Список сообщений сессии
        
    Returns:
        Объединенный текст сессии (только user сообщения)
    """
    if not session_messages:
        return ""
    
    # Объединяем содержимое только сообщений пользователя
    content_parts = []
    for msg in session_messages:
        if msg.role == "user" and msg.content.strip():
            # Фильтруем копипаст для промптов
            if not is_copy_paste_content(msg.content):
                content_parts.append(msg.content.strip())
    
    return " ".join(content_parts)


def extract_session_content_for_search(session_messages: List[Message]) -> str:
    """
    Извлекает текстовое содержимое сессии для поиска ключевых слов (все сообщения)
    
    Args:
        session_messages: Список сообщений сессии
        
    Returns:
        Объединенный текст сессии (все сообщения для поиска)
    """
    if not session_messages:
        return ""
    
    # Объединяем содержимое всех сообщений для поиска ключевых слов
    content_parts = []
    for msg in session_messages:
        if msg.content.strip():
            # Фильтруем копипаст при поиске ключевых слов
            if not is_copy_paste_content(msg.content):
                content_parts.append(msg.content.strip())
    
    return " ".join(content_parts)


def get_session_summary(session_messages: List[Message]) -> Dict[str, any]:
    """
    Получает краткую сводку по сессии
    
    Args:
        session_messages: Список сообщений сессии
        
    Returns:
        Словарь с информацией о сессии
    """
    if not session_messages:
        return {
            'message_count': 0,
            'content_length': 0,
            'has_personal_info': False,
            'session_id': None
        }
    
    content = extract_session_content(session_messages)
    session_id = session_messages[0].session_id if session_messages else None
    
    # Простая проверка на личную информацию
    personal_markers = ['я', 'мой', 'моя', 'мое', 'мы', 'наш', 'наша', 'свой', 'своя']
    has_personal_info = any(marker in content.lower() for marker in personal_markers)
    
    return {
        'message_count': len(session_messages),
        'content_length': len(content),
        'has_personal_info': has_personal_info,
        'session_id': session_id,
        'content_preview': content[:100] + "..." if len(content) > 100 else content
    }
