# src/submit/storage/module.py

from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../..', 'memory_aij2025'))
from models import Message
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'submit'))
from core.interfaces import IStorage, ProcessingResult

from .memory_storage import MemoryStorage
from .message_filter import MessageFilter
from .filters.session_grouper import SessionGrouper
from .filters.relevance_filter import RelevanceFilter


class StorageModule(IStorage):
    """Модуль управления хранилищем и сессиями"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory = MemoryStorage()
        self.filter = MessageFilter()
        self.grouper = SessionGrouper()
        self.relevance = RelevanceFilter(config)
    
    # === Методы интерфейса IStorage ===
    
    def add_messages(self, dialogue_id: str, messages: List[Message]) -> ProcessingResult:
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
    
    def get_messages(self, dialogue_id: str) -> ProcessingResult:
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
    
    def clear(self, dialogue_id: str) -> ProcessingResult:
        """Очищает данные диалога"""
        try:
            self.memory.clear_dialogue_memory(dialogue_id)
            self.grouper.clear_dialogue_sessions(dialogue_id)
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
    
    # === Новые методы для работы с вопросами и RAG ===
    
    def get_filtered_sessions_for_question(self, dialogue_id: str, question: str, 
                                          max_sessions: int = 5) -> ProcessingResult:
        """
        Получает отфильтрованные релевантные сессии для ответа на вопрос
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            max_sessions: Максимальное количество сессий
            
        Returns:
            ProcessingResult с релевантными сессиями
        """
        try:
            # Получаем все сообщения
            messages = self.memory.get_memory(dialogue_id)
            
            # Группируем по сессиям
            all_sessions = self.grouper.group_messages_by_sessions(messages, dialogue_id)
            
            # Находим только релевантные сессии для вопроса
            relevant_sessions = self.relevance.find_relevant_sessions(
                question, all_sessions, max_sessions
            )
            
            # Дополнительная фильтрация контекста под вопрос
            filtered_sessions = {}
            for session_id, session_messages in relevant_sessions.items():
                filtered_msgs = self.filter_for_question_context(session_messages, question)
                if filtered_msgs:
                    filtered_sessions[session_id] = filtered_msgs
            
            return ProcessingResult(
                success=True,
                data=filtered_sessions,
                metadata={
                    'total_sessions': len(all_sessions),
                    'relevant_sessions': len(filtered_sessions),
                    'question': question
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def get_session_content_for_prompt(self, session_messages: List[Message]) -> str:
        """
        Извлекает чистый текст из сессии для промпта
        Убирает копипаст и технический контент
        
        Args:
            session_messages: Сообщения сессии
            
        Returns:
            Очищенный текст для промпта
        """
        from .filters.session_grouper import extract_session_content
        return extract_session_content(session_messages)
    
    def filter_for_question_context(self, messages: List[Message], question: str) -> List[Message]:
        """
        Специальная фильтрация сообщений для контекста вопроса
        Более агрессивно убирает нерелевантное
        
        Args:
            messages: Список сообщений
            question: Вопрос пользователя
            
        Returns:
            Отфильтрованные сообщения
        """
        from .filters.message_cleaner import is_copy_paste_content, is_technical_content
        
        filtered = []
        question_words = set(question.lower().split())
        
        for msg in messages:
            # Проверяем кэш фильтрации
            cache_key = f"{msg.content[:50]}_{question[:20]}"
            cached = self.memory.check_cache(cache_key)
            
            if cached is not None:
                if cached:
                    filtered.append(msg)
                continue
            
            # Фильтруем
            should_include = True
            
            # Пропускаем если это явный копипаст без связи с вопросом
            if is_copy_paste_content(msg.content):
                content_words = set(msg.content.lower().split())
                # Но оставляем если есть пересечение с вопросом
                if not question_words & content_words:
                    should_include = False
            
            # Пропускаем технический контент без связи с вопросом
            if should_include and is_technical_content(msg.content):
                content_words = set(msg.content.lower().split())
                if not question_words & content_words:
                    should_include = False
            
            # Кэшируем результат
            self.memory.add_to_cache(cache_key, should_include)
            
            if should_include:
                filtered.append(msg)
        
        return filtered
    
    def get_relevance_scores(self, dialogue_id: str, question: str) -> ProcessingResult:
        """
        Получает оценки релевантности всех сессий для вопроса
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            
        Returns:
            ProcessingResult со scores для каждой сессии
        """
        try:
            # Получаем все сессии
            messages = self.memory.get_memory(dialogue_id)
            sessions = self.grouper.group_messages_by_sessions(messages, dialogue_id)
            
            # Получаем scores
            scores = self.relevance.get_relevance_scores(question, sessions)
            
            return ProcessingResult(
                success=True,
                data=scores,
                metadata={
                    'sessions_count': len(sessions),
                    'question': question
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def get_session_stats(self, dialogue_id: str) -> ProcessingResult:
        """
        Получает статистику по сессиям диалога
        
        Returns:
            ProcessingResult со статистикой
        """
        try:
            stats = self.memory.get_session_stats(dialogue_id)
            return ProcessingResult(
                success=True,
                data=stats,
                metadata={'dialogue_id': dialogue_id}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def get_memory_stats(self, dialogue_id: str) -> ProcessingResult:
        """
        Получает статистику по памяти диалога
        
        Returns:
            ProcessingResult со статистикой памяти
        """
        try:
            stats = self.memory.get_memory_stats(dialogue_id)
            return ProcessingResult(
                success=True,
                data=stats,
                metadata={'dialogue_id': dialogue_id}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )