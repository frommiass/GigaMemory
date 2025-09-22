"""
Модуль для поиска релевантных сессий по вопросам
Интегрируется с классификатором вопросов и системой тем
"""
from typing import List, Dict, Tuple, Optional, Set
from models import Message

from ..questions.classifier import QuestionClassifier
from ..questions.confidence import calculate_confidence_with_threshold
from .keyword_matcher import KeywordMatcher
from .session_grouper import extract_session_content


class RelevanceFilter:
    """Класс для поиска релевантных сессий"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.classifier = QuestionClassifier()
        self.keyword_matcher = KeywordMatcher()
    
    def _get_default_config(self):
        """Получает конфигурацию по умолчанию без циклического импорта"""
        class DefaultConfig:
            classification_confidence_threshold = 0.7
            max_relevant_sessions = 5
            min_relevant_sessions = 1
        
        return DefaultConfig()
    
    def find_relevant_sessions(self, question: str, sessions: Dict[str, List[Message]], 
                              max_sessions: int = None) -> Dict[str, List[Message]]:
        """
        Находит релевантные сессии для вопроса
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий {session_id: [messages]}
            max_sessions: Максимальное количество сессий (по умолчанию из конфига)
            
        Returns:
            Словарь релевантных сессий
        """
        if not question or not sessions:
            return {}
        
        max_sessions = max_sessions or self.config.max_relevant_sessions
        
        # Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        # Если confidence низкий, используем общий поиск
        if confidence < self.config.classification_confidence_threshold:
            return self._find_relevant_sessions_general(question, sessions, max_sessions)
        
        # Если тема определена, используем тематический поиск
        if topic:
            return self._find_relevant_sessions_by_topic(topic, question, sessions, max_sessions)
        
        # Fallback на общий поиск
        return self._find_relevant_sessions_general(question, sessions, max_sessions)
    
    def _find_relevant_sessions_by_topic(self, topic: str, question: str, 
                                       sessions: Dict[str, List[Message]], 
                                       max_sessions: int) -> Dict[str, List[Message]]:
        """
        Находит релевантные сессии по теме
        
        Args:
            topic: Название темы
            question: Вопрос пользователя
            sessions: Словарь сессий
            max_sessions: Максимальное количество сессий
            
        Returns:
            Словарь релевантных сессий
        """
        # Получаем сессии по теме
        topic_sessions = self.keyword_matcher.find_sessions_by_topic(sessions, topic)
        
        if not topic_sessions:
            return {}
        
        # Ранжируем сессии по релевантности
        ranked_sessions = self.keyword_matcher.get_top_relevant_sessions_by_topic(
            topic_sessions, topic, max_sessions
        )
        
        # Возвращаем топ сессий
        result = {}
        for session_id, score in ranked_sessions:
            if session_id in topic_sessions:
                result[session_id] = topic_sessions[session_id]
        
        return result
    
    def _find_relevant_sessions_general(self, question: str, sessions: Dict[str, List[Message]], 
                                      max_sessions: int) -> Dict[str, List[Message]]:
        """
        Находит релевантные сессии общим поиском
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            max_sessions: Максимальное количество сессий
            
        Returns:
            Словарь релевантных сессий
        """
        # Извлекаем ключевые слова из вопроса
        question_keywords = self._extract_question_keywords(question)
        
        if not question_keywords:
            # Если не удалось извлечь ключевые слова, возвращаем все сессии
            return dict(list(sessions.items())[:max_sessions])
        
        # Ищем сессии по ключевым словам
        relevant_sessions = self.keyword_matcher.find_sessions_by_keywords(
            sessions, question_keywords
        )
        
        # Ранжируем по релевантности
        ranked_sessions = self.keyword_matcher.get_top_relevant_sessions(
            relevant_sessions, question_keywords, max_sessions
        )
        
        # Возвращаем топ сессий
        result = {}
        for session_id, score in ranked_sessions:
            if session_id in relevant_sessions:
                result[session_id] = relevant_sessions[session_id]
        
        return result
    
    def _extract_question_keywords(self, question: str) -> Set[str]:
        """
        Извлекает ключевые слова из вопроса
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Множество ключевых слов
        """
        if not question:
            return set()
        
        # Получаем все ключевые слова из всех тем
        all_keywords = set()
        for topic_name in self.classifier.get_available_topics():
            topic_keywords = self.keyword_matcher.topic_keywords.get(topic_name, set())
            all_keywords.update(topic_keywords)
        
        # Извлекаем слова из вопроса
        question_words = self.keyword_matcher._extract_words(question.lower())
        
        # Находим пересечение
        return set(question_words) & all_keywords
    
    def get_relevance_scores(self, question: str, sessions: Dict[str, List[Message]]) -> Dict[str, float]:
        """
        Получает оценки релевантности для всех сессий
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            
        Returns:
            Словарь {session_id: relevance_score}
        """
        scores = {}
        
        # Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        for session_id, session_messages in sessions.items():
            session_content = extract_session_content(session_messages)
            
            if topic and confidence >= self.config.classification_confidence_threshold:
                # Тематический поиск
                score = self.keyword_matcher.calculate_topic_relevance(session_content, topic)
            else:
                # Общий поиск
                question_keywords = self._extract_question_keywords(question)
                if question_keywords:
                    score = self.keyword_matcher.calculate_relevance_score(session_content, question_keywords)
                else:
                    score = 0.0
            
            scores[session_id] = score
        
        return scores
    
    def filter_sessions_by_threshold(self, sessions: Dict[str, List[Message]], 
                                   scores: Dict[str, float], 
                                   threshold: float = 0.1) -> Dict[str, List[Message]]:
        """
        Фильтрует сессии по порогу релевантности
        
        Args:
            sessions: Словарь сессий
            scores: Словарь оценок релевантности
            threshold: Минимальный порог релевантности
            
        Returns:
            Отфильтрованные сессии
        """
        filtered_sessions = {}
        
        for session_id, session_messages in sessions.items():
            score = scores.get(session_id, 0.0)
            if score >= threshold:
                filtered_sessions[session_id] = session_messages
        
        return filtered_sessions
    
    def get_session_relevance_info(self, question: str, sessions: Dict[str, List[Message]]) -> Dict[str, Dict]:
        """
        Получает детальную информацию о релевантности сессий
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            
        Returns:
            Словарь с детальной информацией о каждой сессии
        """
        info = {}
        
        # Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        for session_id, session_messages in sessions.items():
            session_content = extract_session_content(session_messages)
            
            # Рассчитываем релевантность
            if topic and confidence >= self.config.classification_confidence_threshold:
                relevance_score = self.keyword_matcher.calculate_topic_relevance(session_content, topic)
                matched_keywords = self.keyword_matcher.get_topic_matches(session_content, topic)
            else:
                question_keywords = self._extract_question_keywords(question)
                relevance_score = self.keyword_matcher.calculate_relevance_score(session_content, question_keywords)
                matched_keywords = self.keyword_matcher.get_keyword_matches(session_content, question_keywords)
            
            info[session_id] = {
                'relevance_score': relevance_score,
                'matched_keywords': matched_keywords,
                'topic': topic,
                'confidence': confidence,
                'content_length': len(session_content),
                'message_count': len(session_messages)
            }
        
        return info


# Глобальный экземпляр для удобства использования
relevance_filter = RelevanceFilter()


def find_relevant_sessions(question: str, sessions: Dict[str, List[Message]], 
                          max_sessions: int = None) -> Dict[str, List[Message]]:
    """
    Простая функция для поиска релевантных сессий
    (для обратной совместимости)
    
    Args:
        question: Вопрос пользователя
        sessions: Словарь сессий
        max_sessions: Максимальное количество сессий
        
    Returns:
        Словарь релевантных сессий
    """
    return relevance_filter.find_relevant_sessions(question, sessions, max_sessions)
