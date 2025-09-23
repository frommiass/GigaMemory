"""
Алгоритмы подсчета релевантности для ранжирования сессий
"""
import math
from typing import Dict, List, Tuple, Set
from models import Message

from ..rag.config import DEFAULT_CONFIG


class RelevanceScorer:
    """Класс для расчета релевантности сессий"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
    
    def calculate_session_score(self, session_content: str, question: str, 
                              matched_keywords: Set[str], session_metadata: Dict = None) -> float:
        """
        Рассчитывает общий счет релевантности сессии
        
        Args:
            session_content: Содержимое сессии
            question: Вопрос пользователя
            matched_keywords: Найденные ключевые слова
            session_metadata: Метаданные сессии (время, длина и т.д.)
            
        Returns:
            Оценка релевантности от 0.0 до 1.0
        """
        if not session_content or not question:
            return 0.0
        
        # Базовый счет по ключевым словам
        keyword_score = self._calculate_keyword_score(session_content, matched_keywords)
        
        # Счет по длине сессии
        length_score = self._calculate_length_score(session_content)
        
        # Счет по релевантности к вопросу
        question_relevance_score = self._calculate_question_relevance_score(session_content, question)
        
        # Временной фактор (если есть метаданные)
        recency_score = self._calculate_recency_score(session_metadata) if session_metadata else 0.5
        
        # Взвешенная сумма
        total_score = (
            keyword_score * self.config.keyword_weight +
            length_score * self.config.session_length_weight +
            question_relevance_score * 0.4 +  # Вес релевантности к вопросу
            recency_score * self.config.recency_weight
        )
        
        # Нормализация
        return min(1.0, max(0.0, total_score))
    
    def _calculate_keyword_score(self, content: str, matched_keywords: Set[str]) -> float:
        """
        Рассчитывает счет по ключевым словам
        
        Args:
            content: Содержимое сессии
            matched_keywords: Найденные ключевые слова
            
        Returns:
            Оценка от 0.0 до 1.0
        """
        if not matched_keywords:
            return 0.0
        
        # Подсчитываем частоту ключевых слов
        content_lower = content.lower()
        words = content_lower.split()
        
        if not words:
            return 0.0
        
        keyword_count = sum(1 for word in words if word in matched_keywords)
        total_words = len(words)
        
        # Базовая частота
        frequency = keyword_count / total_words
        
        # Бонус за уникальные ключевые слова
        matched_keywords_set = set(matched_keywords) if isinstance(matched_keywords, list) else matched_keywords
        unique_keywords = len(set(words) & matched_keywords_set)
        uniqueness_bonus = unique_keywords / len(matched_keywords_set) if matched_keywords_set else 0
        
        # Итоговый счет
        score = frequency * 0.7 + uniqueness_bonus * 0.3
        
        return min(1.0, score)
    
    def _calculate_length_score(self, content: str) -> float:
        """
        Рассчитывает счет по длине сессии
        
        Args:
            content: Содержимое сессии
            
        Returns:
            Оценка от 0.0 до 1.0
        """
        length = len(content)
        
        # Оптимальная длина сессии (100-500 символов)
        if 100 <= length <= 500:
            return 1.0
        elif length < 100:
            # Слишком короткие сессии получают штраф
            return length / 100
        else:
            # Слишком длинные сессии получают штраф
            if length > 2000:
                return 0.3
            else:
                return 1.0 - (length - 500) / 1500
    
    def _calculate_question_relevance_score(self, content: str, question: str) -> float:
        """
        Рассчитывает релевантность к вопросу
        
        Args:
            content: Содержимое сессии
            question: Вопрос пользователя
            
        Returns:
            Оценка от 0.0 до 1.0
        """
        if not content or not question:
            return 0.0
        
        # Извлекаем слова из вопроса и контента
        question_words = set(self._extract_words(question.lower()))
        content_words = set(self._extract_words(content.lower()))
        
        if not question_words or not content_words:
            return 0.0
        
        # Пересечение слов
        common_words = question_words & content_words
        
        # Базовый счет по пересечению
        word_overlap = len(common_words) / len(question_words)
        
        # Бонус за важные слова (длинные слова важнее)
        important_words = [word for word in common_words if len(word) > 3]
        importance_bonus = len(important_words) / len(question_words) if question_words else 0
        
        # Итоговый счет
        score = word_overlap * 0.7 + importance_bonus * 0.3
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, session_metadata: Dict) -> float:
        """
        Рассчитывает временной фактор (новые сессии важнее)
        
        Args:
            session_metadata: Метаданные сессии
            
        Returns:
            Оценка от 0.0 до 1.0
        """
        if not session_metadata:
            return 0.5
        
        # Если есть информация о времени
        if 'session_number' in session_metadata:
            session_num = session_metadata['session_number']
            # Более поздние сессии получают более высокий счет
            return min(1.0, session_num / 10.0)  # Нормализация до 10 сессий
        
        # Если есть timestamp
        if 'timestamp' in session_metadata:
            # Здесь можно добавить логику на основе времени
            # Пока возвращаем нейтральное значение
            return 0.5
        
        return 0.5
    
    def _extract_words(self, text: str) -> List[str]:
        """
        Извлекает слова из текста
        
        Args:
            text: Текст для обработки
            
        Returns:
            Список слов в нижнем регистре
        """
        import re
        # Извлекаем слова (только буквы)
        words = re.findall(r'\b[а-яёa-z]+\b', text.lower())
        return words
    
    def calculate_topic_specific_score(self, content: str, topic: str, 
                                     matched_keywords: Set[str]) -> float:
        """
        Рассчитывает тематический счет
        
        Args:
            content: Содержимое сессии
            topic: Название темы
            matched_keywords: Найденные ключевые слова темы
            
        Returns:
            Оценка от 0.0 до 1.0
        """
        if not content or not topic or not matched_keywords:
            return 0.0
        
        # Базовый счет по ключевым словам темы
        keyword_score = self._calculate_keyword_score(content, matched_keywords)
        
        # Бонус за тематическую специфику
        topic_bonus = self._calculate_topic_bonus(content, topic)
        
        # Итоговый счет
        score = keyword_score * 0.8 + topic_bonus * 0.2
        
        return min(1.0, score)
    
    def _calculate_topic_bonus(self, content: str, topic: str) -> float:
        """
        Рассчитывает бонус за тематическую специфику
        
        Args:
            content: Содержимое сессии
            topic: Название темы
            
        Returns:
            Бонус от 0.0 до 1.0
        """
        # Здесь можно добавить специфичную логику для каждой темы
        # Пока возвращаем базовый бонус
        return 0.5
    
    def rank_sessions(self, sessions: Dict[str, List[Message]], 
                     question: str, matched_keywords: Dict[str, Set[str]]) -> List[Tuple[str, float]]:
        """
        Ранжирует сессии по релевантности
        
        Args:
            sessions: Словарь сессий
            question: Вопрос пользователя
            matched_keywords: Словарь {session_id: matched_keywords}
            
        Returns:
            Список кортежей (session_id, score) отсортированный по убыванию
        """
        session_scores = []
        
        for session_id, session_messages in sessions.items():
            # Извлекаем содержимое сессии
            session_content = " ".join(msg.content for msg in session_messages if msg.role == "user")
            
            # Получаем метаданные сессии
            session_metadata = {
                'message_count': len(session_messages),
                'content_length': len(session_content)
            }
            
            # Рассчитываем счет
            keywords = matched_keywords.get(session_id, set())
            score = self.calculate_session_score(
                session_content, question, keywords, session_metadata
            )
            
            session_scores.append((session_id, score))
        
        # Сортируем по убыванию счета
        session_scores.sort(key=lambda x: x[1], reverse=True)
        
        return session_scores


# Глобальный экземпляр для удобства использования
relevance_scorer = RelevanceScorer()


def calculate_session_relevance(session_content: str, question: str, 
                              matched_keywords: Set[str]) -> float:
    """
    Простая функция для расчета релевантности сессии
    (для обратной совместимости)
    
    Args:
        session_content: Содержимое сессии
        question: Вопрос пользователя
        matched_keywords: Найденные ключевые слова
        
    Returns:
        Оценка релевантности от 0.0 до 1.0
    """
    return relevance_scorer.calculate_session_score(session_content, question, matched_keywords)
