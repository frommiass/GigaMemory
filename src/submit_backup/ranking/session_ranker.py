"""
Модуль для ранжирования сессий по релевантности
Интегрируется с системой скоринга и фильтрации
"""
from typing import Dict, List, Tuple, Optional
from models import Message

from ..questions.classifier import QuestionClassifier
from ..rag.config import DEFAULT_CONFIG
from .scorer import RelevanceScorer
from ..filters.keyword_matcher import KeywordMatcher
from ..filters.session_grouper import extract_session_content


class SessionRanker:
    """Класс для ранжирования сессий"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.scorer = RelevanceScorer(config)
        self.classifier = QuestionClassifier()
        self.keyword_matcher = KeywordMatcher()
    
    def rank_sessions(self, question: str, sessions: Dict[str, List[Message]], 
                     top_k: int = None) -> List[Tuple[str, float]]:
        """
        Ранжирует сессии по релевантности к вопросу
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий {session_id: [messages]}
            top_k: Количество топ сессий (по умолчанию из конфига)
            
        Returns:
            Список кортежей (session_id, score) отсортированный по убыванию
        """
        if not question or not sessions:
            return []
        
        top_k = top_k or self.config.max_relevant_sessions
        
        # Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        # Получаем ключевые слова для поиска
        if topic and confidence >= self.config.classification_confidence_threshold:
            # Тематический поиск
            keywords = self.keyword_matcher.topic_keywords.get(topic, set())
        else:
            # Общий поиск
            keywords = self._extract_question_keywords(question)
        
        # Рассчитываем релевантность для каждой сессии
        session_scores = []
        
        for session_id, session_messages in sessions.items():
            session_content = extract_session_content(session_messages)
            
            # Находим совпадающие ключевые слова
            matched_keywords = self.keyword_matcher.get_keyword_matches(session_content, keywords)
            
            # Рассчитываем счет релевантности
            if topic and confidence >= self.config.classification_confidence_threshold:
                score = self.scorer.calculate_topic_specific_score(
                    session_content, topic, matched_keywords
                )
            else:
                score = self.scorer.calculate_session_score(
                    session_content, question, matched_keywords
                )
            
            session_scores.append((session_id, score))
        
        # Сортируем по убыванию счета
        session_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-K сессий
        return session_scores[:top_k]
    
    def rank_sessions_with_metadata(self, question: str, sessions: Dict[str, List[Message]], 
                                  top_k: int = None) -> List[Dict]:
        """
        Ранжирует сессии с дополнительными метаданными
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            top_k: Количество топ сессий
            
        Returns:
            Список словарей с информацией о сессиях
        """
        if not question or not sessions:
            return []
        
        top_k = top_k or self.config.max_relevant_sessions
        
        # Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        # Получаем ключевые слова
        if topic and confidence >= self.config.classification_confidence_threshold:
            keywords = self.keyword_matcher.topic_keywords.get(topic, set())
        else:
            keywords = self._extract_question_keywords(question)
        
        # Рассчитываем релевантность для каждой сессии
        session_info = []
        
        for session_id, session_messages in sessions.items():
            session_content = extract_session_content(session_messages)
            
            # Находим совпадающие ключевые слова
            matched_keywords = self.keyword_matcher.get_keyword_matches(session_content, keywords)
            
            # Рассчитываем счет релевантности
            if topic and confidence >= self.config.classification_confidence_threshold:
                score = self.scorer.calculate_topic_specific_score(
                    session_content, topic, matched_keywords
                )
            else:
                score = self.scorer.calculate_session_score(
                    session_content, question, matched_keywords
                )
            
            # Собираем метаданные
            session_info.append({
                'session_id': session_id,
                'score': score,
                'matched_keywords': list(matched_keywords),
                'content_length': len(session_content),
                'message_count': len(session_messages),
                'topic': topic,
                'confidence': confidence,
                'content_preview': session_content[:100] + "..." if len(session_content) > 100 else session_content
            })
        
        # Сортируем по убыванию счета
        session_info.sort(key=lambda x: x['score'], reverse=True)
        
        # Возвращаем топ-K сессий
        return session_info[:top_k]
    
    def get_top_sessions(self, question: str, sessions: Dict[str, List[Message]], 
                        top_k: int = None) -> Dict[str, List[Message]]:
        """
        Получает топ-K наиболее релевантных сессий
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            top_k: Количество топ сессий
            
        Returns:
            Словарь топ сессий
        """
        ranked_sessions = self.rank_sessions(question, sessions, top_k)
        
        result = {}
        for session_id, score in ranked_sessions:
            if session_id in sessions:
                result[session_id] = sessions[session_id]
        
        return result
    
    def filter_sessions_by_score(self, question: str, sessions: Dict[str, List[Message]], 
                               min_score: float = 0.1) -> Dict[str, List[Message]]:
        """
        Фильтрует сессии по минимальному счету релевантности
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            min_score: Минимальный счет релевантности
            
        Returns:
            Отфильтрованные сессии
        """
        ranked_sessions = self.rank_sessions(question, sessions)
        
        result = {}
        for session_id, score in ranked_sessions:
            if score >= min_score and session_id in sessions:
                result[session_id] = sessions[session_id]
        
        return result
    
    def get_session_ranking_stats(self, question: str, sessions: Dict[str, List[Message]]) -> Dict:
        """
        Получает статистику ранжирования сессий
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            
        Returns:
            Словарь со статистикой
        """
        if not sessions:
            return {
                'total_sessions': 0,
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'high_relevance_count': 0
            }
        
        ranked_sessions = self.rank_sessions(question, sessions)
        
        if not ranked_sessions:
            return {
                'total_sessions': len(sessions),
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'high_relevance_count': 0
            }
        
        scores = [score for _, score in ranked_sessions]
        
        return {
            'total_sessions': len(sessions),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'high_relevance_count': sum(1 for score in scores if score >= 0.5),
            'top_5_scores': scores[:5]
        }
    
    def _extract_question_keywords(self, question: str) -> set:
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


# Глобальный экземпляр для удобства использования
session_ranker = SessionRanker()


def rank_sessions(question: str, sessions: Dict[str, List[Message]], 
                 top_k: int = None) -> List[Tuple[str, float]]:
    """
    Простая функция для ранжирования сессий
    (для обратной совместимости)
    
    Args:
        question: Вопрос пользователя
        sessions: Словарь сессий
        top_k: Количество топ сессий
        
    Returns:
        Список кортежей (session_id, score)
    """
    return session_ranker.rank_sessions(question, sessions, top_k)
