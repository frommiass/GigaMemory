# src/submit/storage/filters/session_grouper.py
"""
Объединенный модуль для группировки сообщений по сессиям
Содержит функционал из SessionGrouper и SessionManager
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from models import Message
from .message_cleaner import is_copy_paste_content


class SessionGrouper:
    """Объединенный класс для группировки сообщений по сессиям и управления ими"""
    
    def __init__(self):
        # Счетчики сессий для каждого диалога
        self.session_counters = defaultdict(int)
        
        # Реестр зарегистрированных сессий
        self.registered_sessions = defaultdict(set)
        
        # Группированные сообщения по диалогам и сессиям
        self.grouped_sessions = defaultdict(lambda: defaultdict(list))
        
        # Детальная информация о сессиях
        self.session_info = defaultdict(dict)
    
    # === Методы группировки ===
    
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
                if hasattr(msg, 'session_id') and msg.session_id:
                    session_id = msg.session_id
                else:
                    # Создаем новую сессию
                    session_id = str(self.increment_session(dialogue_id))
                
                # Регистрируем сессию
                self.register_session(dialogue_id, session_id)
                
                # Создаем новое сообщение с session_id
                marked_msg = Message(
                    role=msg.role,
                    content=msg.content,
                    session_id=session_id
                )
                
                sessions[session_id].append(marked_msg)
            elif msg.role == "assistant":
                # Добавляем ответы ассистента к последней сессии
                if sessions:
                    last_session_id = list(sessions.keys())[-1]
                    sessions[last_session_id].append(msg)
        
        # Сохраняем в группированном виде
        self.grouped_sessions[dialogue_id] = sessions
        
        # Обновляем информацию о сессиях
        for session_id, session_messages in sessions.items():
            self.add_session_info(dialogue_id, session_id, len(session_messages))
        
        return dict(sessions)
    
    # === Методы управления сессиями ===
    
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
                if not self.registered_sessions[dialogue_id]:
                    self.session_counters[dialogue_id] += 1
    
    def get_current_session(self, dialogue_id: str) -> int:
        """Возвращает текущий номер сессии"""
        return self.session_counters[dialogue_id]
    
    def get_session_count(self, dialogue_id: str) -> int:
        """Возвращает количество сессий в диалоге"""
        return len(self.registered_sessions[dialogue_id])
    
    def get_session_ids(self, dialogue_id: str) -> List[str]:
        """Возвращает список ID сессий диалога"""
        return sorted(list(self.registered_sessions[dialogue_id]))
    
    # === Методы работы с данными сессий ===
    
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
    
    def add_session_info(self, dialogue_id: str, session_id: str, message_count: int = 0) -> None:
        """Добавляет информацию о сессии"""
        self.session_info[dialogue_id][session_id] = {
            'session_id': session_id,
            'dialogue_id': dialogue_id,
            'message_count': message_count
        }
    
    def get_session_info(self, dialogue_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о конкретной сессии"""
        return self.session_info[dialogue_id].get(session_id)
    
    # === Методы очистки ===
    
    def clear_dialogue_sessions(self, dialogue_id: str) -> None:
        """Очищает все сессии диалога"""
        self.session_counters[dialogue_id] = 0
        self.registered_sessions[dialogue_id].clear()
        self.grouped_sessions[dialogue_id] = defaultdict(list)
        self.session_info[dialogue_id].clear()
    
    # === Методы статистики ===
    
    def get_session_stats(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Получает статистику по сессиям диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь со статистикой
        """
        sessions = self.grouped_sessions.get(dialogue_id, {})
        total_messages = sum(len(msgs) for msgs in sessions.values())
        session_ids = self.get_session_ids(dialogue_id)
        
        return {
            'dialogue_id': dialogue_id,
            'total_sessions': len(sessions),
            'total_messages': total_messages,
            'avg_messages_per_session': total_messages / len(sessions) if sessions else 0,
            'session_ids': session_ids
        }
    
    def get_global_stats(self) -> Dict[str, int]:
        """Возвращает глобальную статистику по всем диалогам"""
        total_dialogues = len(self.session_counters)
        total_sessions = sum(self.session_counters.values())
        
        return {
            'total_dialogues': total_dialogues,
            'total_sessions': total_sessions,
            'avg_sessions_per_dialogue': total_sessions / total_dialogues if total_dialogues > 0 else 0
        }
    
    def is_session_registered(self, dialogue_id: str, session_id: str) -> bool:
        """Проверяет, зарегистрирована ли сессия"""
        return session_id in self.registered_sessions[dialogue_id]


# === Вспомогательные функции ===

def group_messages_by_sessions_simple(messages: List[Message], dialogue_id: str) -> Dict[str, List[Message]]:
    """
    Простая функция для группировки сообщений по сессиям
    (для обратной совместимости)
    """
    grouper = SessionGrouper()
    return grouper.group_messages_by_sessions(messages, dialogue_id)


def extract_session_content(session_messages: List[Message]) -> str:
    """
    Извлекает текстовое содержимое сессии для промпта (только сообщения пользователя)
    Фильтрует копипаст
    """
    if not session_messages:
        return ""
    
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
    """
    if not session_messages:
        return ""
    
    content_parts = []
    for msg in session_messages:
        if msg.content.strip():
            # Фильтруем копипаст при поиске ключевых слов
            if not is_copy_paste_content(msg.content):
                content_parts.append(msg.content.strip())
    
    return " ".join(content_parts)


def get_session_summary(session_messages: List[Message]) -> Dict[str, Any]:
    """
    Получает краткую сводку по сессии
    """
    if not session_messages:
        return {
            'message_count': 0,
            'content_length': 0,
            'has_personal_info': False,
            'session_id': None
        }
    
    content = extract_session_content(session_messages)
    session_id = getattr(session_messages[0], 'session_id', None) if session_messages else None
    
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