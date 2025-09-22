"""
Центральный модуль для фильтрации сообщений от копипаста и технического контента.

Этот модуль содержит всю логику фильтрации, которая используется всеми компонентами системы.
Принцип: "Фильтруй один раз, используй везде"
"""
from typing import List, Dict, Tuple
from models import Message, Dialog

from ..filters.message_cleaner import (
    is_copy_paste_content,
    is_personal_message,
    is_technical_content,
    get_message_quality_score
)


class MessageFilter:
    """Центральный класс для фильтрации сообщений"""
    
    def __init__(self):
        """Инициализация фильтра сообщений"""
        pass
    
    def filter_dialogs(self, dialogs: List[Dialog]) -> List[Dialog]:
        """
        Фильтрует диалоги, удаляя копипаст и технический контент
        
        Args:
            dialogs: Список диалогов для фильтрации
            
        Returns:
            Список отфильтрованных диалогов
        """
        filtered_dialogs = []
        
        for dialog in dialogs:
            # Фильтруем сообщения в каждой сессии диалога
            filtered_sessions = []
            for session in dialog.sessions:
                filtered_messages = self.filter_messages(session.messages)
                
                # Создаем новую сессию с отфильтрованными сообщениями
                if filtered_messages:  # Добавляем сессию только если в ней остались сообщения
                    from models import Session
                    filtered_session = Session(
                        id=session.id,
                        messages=filtered_messages
                    )
                    filtered_sessions.append(filtered_session)
            
            # Создаем новый диалог с отфильтрованными сессиями
            if filtered_sessions:  # Добавляем диалог только если в нем остались сессии
                filtered_dialog = Dialog(
                    id=dialog.id,
                    question=dialog.question,
                    sessions=filtered_sessions
                )
                filtered_dialogs.append(filtered_dialog)
        
        return filtered_dialogs
    
    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """
        Фильтрует сообщения, удаляя копипаст и технический контент
        
        Args:
            messages: Список сообщений для фильтрации
            
        Returns:
            Список отфильтрованных сообщений
        """
        if not messages:
            return []
        
        filtered_messages = []
        previous_was_copypaste = False
        
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                # Проверяем, не является ли сообщение копипастом
                is_copypaste = is_copy_paste_content(msg.content)
                
                # Если предыдущее сообщение было копипастом, то и это тоже копипаст
                if previous_was_copypaste:
                    is_copypaste = True
                
                # Фильтруем копипаст, но сохраняем личную информацию
                if not is_copypaste and is_personal_message(msg.content):
                    filtered_messages.append(msg)
                    previous_was_copypaste = False
                else:
                    previous_was_copypaste = True
            else:
                # Сохраняем сообщения ассистента
                filtered_messages.append(msg)
                previous_was_copypaste = False
        
        return filtered_messages
    
    def filter_messages_simple(self, messages: List[Message]) -> List[Message]:
        """
        Простая фильтрация сообщений (только удаление копипаста)
        
        Args:
            messages: Список сообщений для фильтрации
            
        Returns:
            Список отфильтрованных сообщений
        """
        if not messages:
            return []
        
        filtered_messages = []
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                # Проверяем, не является ли сообщение копипастом
                if not is_copy_paste_content(msg.content):
                    filtered_messages.append(msg)
            else:
                # Сохраняем сообщения ассистента
                filtered_messages.append(msg)
        
        return filtered_messages
    
    def get_message_analysis(self, messages: List[Message]) -> Dict[str, any]:
        """
        Анализирует сообщения и возвращает статистику фильтрации
        
        Args:
            messages: Список сообщений для анализа
            
        Returns:
            Словарь с анализом сообщений
        """
        if not messages:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'filtered_messages': 0,
                'copypaste_messages': 0,
                'technical_messages': 0,
                'personal_messages': 0,
                'filter_ratio': 0.0
            }
        
        total_messages = len(messages)
        user_messages = sum(1 for msg in messages if msg.role == "user")
        
        copypaste_count = 0
        technical_count = 0
        personal_count = 0
        
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                if is_copy_paste_content(msg.content):
                    copypaste_count += 1
                elif is_technical_content(msg.content):
                    technical_count += 1
                elif is_personal_message(msg.content):
                    personal_count += 1
        
        filtered_messages = self.filter_messages(messages)
        filtered_count = len(filtered_messages)
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'filtered_messages': filtered_count,
            'copypaste_messages': copypaste_count,
            'technical_messages': technical_count,
            'personal_messages': personal_count,
            'filter_ratio': (user_messages - filtered_count) / user_messages if user_messages > 0 else 0.0
        }
    
    def get_message_quality_analysis(self, messages: List[Message]) -> List[Dict[str, any]]:
        """
        Анализирует качество каждого сообщения
        
        Args:
            messages: Список сообщений для анализа
            
        Returns:
            Список словарей с анализом каждого сообщения
        """
        analysis = []
        
        for i, msg in enumerate(messages):
            if msg.role == "user" and msg.content.strip():
                quality_score = get_message_quality_score(msg.content)
                is_copypaste = is_copy_paste_content(msg.content)
                is_technical = is_technical_content(msg.content)
                is_personal = is_personal_message(msg.content)
                
                analysis.append({
                    'index': i,
                    'content_preview': msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    'content_length': len(msg.content),
                    'quality_score': quality_score,
                    'is_copypaste': is_copypaste,
                    'is_technical': is_technical,
                    'is_personal': is_personal,
                    'will_be_filtered': is_copypaste or not is_personal
                })
        
        return analysis


# Глобальный экземпляр для удобства использования
message_filter = MessageFilter()


def filter_dialogs(dialogs: List[Dialog]) -> List[Dialog]:
    """
    Простая функция для фильтрации диалогов
    (для обратной совместимости)
    
    Args:
        dialogs: Список диалогов для фильтрации
        
    Returns:
        Список отфильтрованных диалогов
    """
    return message_filter.filter_dialogs(dialogs)


def filter_messages(messages: List[Message]) -> List[Message]:
    """
    Простая функция для фильтрации сообщений
    (для обратной совместимости)
    
    Args:
        messages: Список сообщений для фильтрации
        
    Returns:
        Список отфильтрованных сообщений
    """
    return message_filter.filter_messages(messages)
