"""
Модуль для поиска по ключевым словам в сообщениях и сессиях
Интегрируется с существующей системой тем из questions/topics.py
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from models import Message

from ..questions.topics import get_all_topics, get_topic_keywords, Topic
from .session_grouper import extract_session_content


class KeywordMatcher:
    """Класс для поиска по ключевым словам"""
    
    def __init__(self):
        self.topics = get_all_topics()
        self.topic_keywords = {name: get_topic_keywords(name) for name in self.topics.keys()}
    
    def find_messages_by_keywords(self, messages: List[Message], keywords: Set[str]) -> List[Message]:
        """
        Находит сообщения, содержащие ключевые слова
        
        Args:
            messages: Список сообщений для поиска
            keywords: Множество ключевых слов
            
        Returns:
            Список сообщений, содержащих ключевые слова
        """
        matching_messages = []
        
        for msg in messages:
            if msg.role == "user" and self._contains_keywords(msg.content, keywords):
                matching_messages.append(msg)
        
        return matching_messages
    
    def find_sessions_by_keywords(self, sessions: Dict[str, List[Message]], keywords: Set[str]) -> Dict[str, List[Message]]:
        """
        Находит сессии, содержащие ключевые слова
        
        Args:
            sessions: Словарь сессий {session_id: [messages]}
            keywords: Множество ключевых слов
            
        Returns:
            Словарь релевантных сессий
        """
        matching_sessions = {}
        
        for session_id, session_messages in sessions.items():
            session_content = extract_session_content(session_messages)
            if self._contains_keywords(session_content, keywords):
                matching_sessions[session_id] = session_messages
        
        return matching_sessions
    
    def find_by_topic(self, messages: List[Message], topic_name: str) -> List[Message]:
        """
        Находит сообщения по теме
        
        Args:
            messages: Список сообщений
            topic_name: Название темы
            
        Returns:
            Список сообщений, относящихся к теме
        """
        if topic_name not in self.topic_keywords:
            return []
        
        keywords = self.topic_keywords[topic_name]
        return self.find_messages_by_keywords(messages, keywords)
    
    def find_sessions_by_topic(self, sessions: Dict[str, List[Message]], topic_name: str) -> Dict[str, List[Message]]:
        """
        Находит сессии по теме
        
        Args:
            sessions: Словарь сессий
            topic_name: Название темы
            
        Returns:
            Словарь релевантных сессий
        """
        if topic_name not in self.topic_keywords:
            return {}
        
        keywords = self.topic_keywords[topic_name]
        return self.find_sessions_by_keywords(sessions, keywords)
    
    def calculate_relevance_score(self, content: str, keywords: Set[str]) -> float:
        """
        Рассчитывает релевантность контента ключевым словам
        
        Args:
            content: Текст для анализа
            keywords: Множество ключевых слов
            
        Returns:
            Оценка релевантности от 0.0 до 1.0
        """
        if not content or not keywords:
            return 0.0
        
        content_lower = content.lower()
        content_words = self._extract_words(content_lower)
        
        if not content_words:
            return 0.0
        
        # Подсчитываем совпадения
        matches = 0
        for word in content_words:
            if word in keywords:
                matches += 1
        
        # Рассчитываем базовую релевантность
        relevance = matches / len(content_words)
        
        # Бонус за уникальные совпадения
        unique_matches = len(set(content_words) & keywords)
        if unique_matches > 0:
            relevance += unique_matches * 0.1
        
        return min(1.0, relevance)
    
    def calculate_topic_relevance(self, content: str, topic_name: str) -> float:
        """
        Рассчитывает релевантность контента теме
        
        Args:
            content: Текст для анализа
            topic_name: Название темы
            
        Returns:
            Оценка релевантности теме от 0.0 до 1.0
        """
        if topic_name not in self.topic_keywords:
            return 0.0
        
        keywords = self.topic_keywords[topic_name]
        return self.calculate_relevance_score(content, keywords)
    
    def get_top_relevant_sessions(self, sessions: Dict[str, List[Message]], 
                                 keywords: Set[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Получает топ-K наиболее релевантных сессий
        
        Args:
            sessions: Словарь сессий
            keywords: Множество ключевых слов
            top_k: Количество топ сессий
            
        Returns:
            Список кортежей (session_id, relevance_score)
        """
        session_scores = []
        
        for session_id, session_messages in sessions.items():
            session_content = extract_session_content(session_messages)
            score = self.calculate_relevance_score(session_content, keywords)
            if score > 0:
                session_scores.append((session_id, score))
        
        # Сортируем по убыванию релевантности
        session_scores.sort(key=lambda x: x[1], reverse=True)
        
        return session_scores[:top_k]
    
    def get_top_relevant_sessions_by_topic(self, sessions: Dict[str, List[Message]], 
                                         topic_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Получает топ-K наиболее релевантных сессий по теме
        
        Args:
            sessions: Словарь сессий
            topic_name: Название темы
            top_k: Количество топ сессий
            
        Returns:
            Список кортежей (session_id, relevance_score)
        """
        if topic_name not in self.topic_keywords:
            return []
        
        keywords = self.topic_keywords[topic_name]
        return self.get_top_relevant_sessions(sessions, keywords, top_k)
    
    def _contains_keywords(self, content: str, keywords: Set[str]) -> bool:
        """
        Проверяет, содержит ли контент ключевые слова
        
        Args:
            content: Текст для проверки
            keywords: Множество ключевых слов
            
        Returns:
            True если найдены ключевые слова
        """
        if not content or not keywords:
            return False
        
        content_lower = content.lower()
        content_words = self._extract_words(content_lower)
        
        return bool(set(content_words) & keywords)
    
    def _extract_words(self, text: str) -> List[str]:
        """
        Извлекает слова из текста
        
        Args:
            text: Текст для обработки
            
        Returns:
            Список слов в нижнем регистре
        """
        # Извлекаем слова (только буквы)
        words = re.findall(r'\b[а-яёa-z]+\b', text.lower())
        return words
    
    def get_keyword_matches(self, content: str, keywords: Set[str]) -> List[str]:
        """
        Получает список найденных ключевых слов в контенте
        
        Args:
            content: Текст для поиска
            keywords: Множество ключевых слов
            
        Returns:
            Список найденных ключевых слов
        """
        if not content or not keywords:
            return []
        
        content_lower = content.lower()
        content_words = set(self._extract_words(content_lower))
        
        return list(content_words & keywords)
    
    def get_topic_matches(self, content: str, topic_name: str) -> List[str]:
        """
        Получает список найденных ключевых слов темы в контенте
        
        Args:
            content: Текст для поиска
            topic_name: Название темы
            
        Returns:
            Список найденных ключевых слов темы
        """
        if topic_name not in self.topic_keywords:
            return []
        
        keywords = self.topic_keywords[topic_name]
        return self.get_keyword_matches(content, keywords)


# Глобальный экземпляр для удобства использования
keyword_matcher = KeywordMatcher()


def find_messages_by_topic(messages: List[Message], topic_name: str) -> List[Message]:
    """
    Простая функция для поиска сообщений по теме
    (для обратной совместимости)
    
    Args:
        messages: Список сообщений
        topic_name: Название темы
        
    Returns:
        Список релевантных сообщений
    """
    return keyword_matcher.find_by_topic(messages, topic_name)


def find_sessions_by_topic(sessions: Dict[str, List[Message]], topic_name: str) -> Dict[str, List[Message]]:
    """
    Простая функция для поиска сессий по теме
    (для обратной совместимости)
    
    Args:
        sessions: Словарь сессий
        topic_name: Название темы
        
    Returns:
        Словарь релевантных сессий
    """
    return keyword_matcher.find_sessions_by_topic(sessions, topic_name)
