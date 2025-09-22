"""
Классификатор вопросов по темам на основе ключевых слов
"""
import re
from typing import Dict, List, Tuple, Optional
from .topics import get_all_topics, get_topic_keywords, Topic
from .confidence import calculate_confidence


class QuestionClassifier:
    """Классификатор вопросов по темам"""
    
    def __init__(self):
        self.topics = get_all_topics()
        self.topic_keywords = {name: get_topic_keywords(name) for name in self.topics.keys()}
    
    def classify_question(self, question: str) -> Tuple[Optional[str], float]:
        """
        Классифицирует вопрос по темам
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Tuple[название_темы, confidence_score]
            Если тема не определена, возвращает (None, 0.0)
        """
        if not question or not question.strip():
            return None, 0.0
        
        question_lower = question.lower()
        question_words = self._extract_words(question_lower)
        
        if not question_words:
            return None, 0.0
        
        # Подсчитываем совпадения для каждой темы
        topic_scores = {}
        
        for topic_name, topic in self.topics.items():
            score = self._calculate_topic_score(question_words, topic_name)
            if score > 0:
                topic_scores[topic_name] = score
        
        if not topic_scores:
            return None, 0.0
        
        # Находим тему с максимальным счетом
        best_topic = max(topic_scores.items(), key=lambda x: x[1])
        topic_name, score = best_topic
        
        # Рассчитываем confidence
        confidence = calculate_confidence(score, len(question_words), topic_scores)
        
        return topic_name, confidence
    
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
    
    def _calculate_topic_score(self, question_words: List[str], topic_name: str) -> float:
        """
        Рассчитывает счет для конкретной темы
        
        Args:
            question_words: Слова из вопроса
            question_words: Список слов из вопроса
            topic_name: Название темы
            
        Returns:
            Счет темы (0.0 если нет совпадений)
        """
        topic_keywords = self.topic_keywords[topic_name]
        topic = self.topics[topic_name]
        
        # Подсчитываем точные совпадения
        exact_matches = 0
        for word in question_words:
            if word in topic_keywords:
                exact_matches += 1
        
        if exact_matches == 0:
            return 0.0
        
        # Рассчитываем базовый счет
        base_score = exact_matches / len(question_words)
        
        # Применяем вес темы
        weighted_score = base_score * topic.weight
        
        return weighted_score
    
    def get_top_topics(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Получает топ-K тем для вопроса
        
        Args:
            question: Вопрос пользователя
            top_k: Количество топ тем
            
        Returns:
            Список кортежей (название_темы, confidence_score)
        """
        if not question or not question.strip():
            return []
        
        question_lower = question.lower()
        question_words = self._extract_words(question_lower)
        
        if not question_words:
            return []
        
        # Подсчитываем совпадения для всех тем
        topic_scores = {}
        
        for topic_name, topic in self.topics.items():
            score = self._calculate_topic_score(question_words, topic_name)
            if score > 0:
                confidence = calculate_confidence(score, len(question_words), {topic_name: score})
                topic_scores[topic_name] = confidence
        
        # Сортируем по убыванию confidence
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_topics[:top_k]
    
    def get_available_topics(self) -> List[str]:
        """
        Получает список доступных тем
        
        Returns:
            Список названий тем
        """
        return list(self.topics.keys())


# Глобальный экземпляр классификатора
classifier = QuestionClassifier()
