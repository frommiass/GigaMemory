"""
Главный RAG движок с логикой фолбэка
Интегрирует все компоненты: классификацию, фильтрацию, ранжирование, промпты
"""
from typing import Dict, List, Tuple, Optional
from models import Message

from .config import DEFAULT_CONFIG
from ..questions.classifier import QuestionClassifier
from ..questions.confidence import calculate_confidence_with_threshold
from ..filters.relevance_filter import RelevanceFilter
from ..filters.session_grouper import SessionGrouper, extract_session_content
from ..ranking.session_ranker import SessionRanker
from ..prompts.topic_prompts import get_topic_prompt
from ..prompts.fallback_prompts import get_fallback_prompt


class RAGEngine:
    """Главный RAG движок для обработки вопросов"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.classifier = QuestionClassifier()
        self.relevance_filter = RelevanceFilter(config)
        self.session_grouper = SessionGrouper()
        self.session_ranker = SessionRanker(config)
    
    def process_question(self, question: str, dialogue_id: str, 
                        all_messages: List[Message]) -> Tuple[str, Dict]:
        """
        Обрабатывает вопрос с использованием RAG системы
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Tuple[ответ, метаданные_обработки]
        """
        if not question or not all_messages:
            return "У меня нет информации для ответа на этот вопрос.", {}
        
        # 1. Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        # 2. Группируем сообщения по сессиям
        sessions = self.session_grouper.group_messages_by_sessions(all_messages, dialogue_id)
        
        # 3. Определяем стратегию обработки
        if confidence >= self.config.classification_confidence_threshold and topic:
            # Тематический RAG с релевантными сессиями
            return self._process_with_topic_rag(question, topic, confidence, sessions)
        else:
            # Fallback на все сессии
            return self._process_with_fallback(question, sessions)
    
    def _process_with_topic_rag(self, question: str, topic: str, confidence: float, 
                               sessions: Dict[str, List[Message]]) -> Tuple[str, Dict]:
        """
        Обрабатывает вопрос с тематическим RAG
        
        Args:
            question: Вопрос пользователя
            topic: Определенная тема
            confidence: Уверенность в классификации
            sessions: Словарь сессий
            
        Returns:
            Tuple[ответ, метаданные]
        """
        # 4. Находим релевантные сессии по теме
        relevant_sessions = self.relevance_filter.find_relevant_sessions(
            question, sessions, self.config.max_relevant_sessions
        )
        
        if not relevant_sessions:
            # Если релевантных сессий нет, используем fallback
            return self._process_with_fallback(question, sessions)
        
        # 5. Ранжируем релевантные сессии
        ranked_sessions = self.session_ranker.rank_sessions(
            question, relevant_sessions, self.config.max_relevant_sessions
        )
        
        # 6. Выбираем топ сессии
        top_session_ids = [session_id for session_id, score in ranked_sessions[:self.config.max_relevant_sessions]]
        top_sessions = {session_id: relevant_sessions[session_id] for session_id in top_session_ids if session_id in relevant_sessions}
        
        # 7. Формируем контекст из топ сессий
        memory_text = self._build_memory_context(top_sessions)
        
        # 8. Генерируем тематический промпт
        prompt = get_topic_prompt(topic, question, memory_text)
        
        # 9. Метаданные для отладки
        metadata = {
            'strategy': 'topic_rag',
            'topic': topic,
            'confidence': confidence,
            'total_sessions': len(sessions),
            'relevant_sessions': len(relevant_sessions),
            'selected_sessions': len(top_sessions),
            'selected_session_ids': list(top_sessions.keys()),
            'memory_length': len(memory_text)
        }
        
        return prompt, metadata
    
    def _process_with_fallback(self, question: str, sessions: Dict[str, List[Message]]) -> Tuple[str, Dict]:
        """
        Обрабатывает вопрос с fallback стратегией (все сессии)
        
        Args:
            question: Вопрос пользователя
            sessions: Словарь сессий
            
        Returns:
            Tuple[ответ, метаданные]
        """
        # 4. Используем все сессии (fallback)
        all_sessions = sessions
        
        # 5. Ранжируем все сессии по релевантности
        ranked_sessions = self.session_ranker.rank_sessions(
            question, all_sessions, self.config.max_relevant_sessions
        )
        
        # 6. Выбираем топ сессии
        top_session_ids = [session_id for session_id, score in ranked_sessions[:self.config.max_relevant_sessions]]
        top_sessions = {session_id: all_sessions[session_id] for session_id in top_session_ids if session_id in all_sessions}
        
        # 7. Формируем контекст из топ сессий
        memory_text = self._build_memory_context(top_sessions)
        
        # 8. Генерируем fallback промпт
        prompt = get_fallback_prompt(question, memory_text)
        
        # 9. Метаданные для отладки
        metadata = {
            'strategy': 'fallback',
            'topic': None,
            'confidence': 0.0,
            'total_sessions': len(sessions),
            'relevant_sessions': len(sessions),
            'selected_sessions': len(top_sessions),
            'selected_session_ids': list(top_sessions.keys()),
            'memory_length': len(memory_text)
        }
        
        return prompt, metadata
    
    def _build_memory_context(self, sessions: Dict[str, List[Message]]) -> str:
        """
        Строит контекст памяти из сессий
        
        Args:
            sessions: Словарь сессий
            
        Returns:
            Текстовый контекст для промпта
        """
        if not sessions:
            return ""
        
        context_parts = []
        
        for session_id, session_messages in sessions.items():
            # Извлекаем содержимое сессии
            session_content = extract_session_content(session_messages)
            
            if session_content.strip():
                # Добавляем маркер сессии только один раз для всей сессии
                session_marker = f"[Сессия {session_id}]"
                context_parts.append(f"{session_marker} {session_content}")
        
        return "\n\n".join(context_parts)
    
    def get_question_analysis(self, question: str, dialogue_id: str, 
                            all_messages: List[Message]) -> Dict:
        """
        Получает детальный анализ вопроса без генерации ответа
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Словарь с анализом
        """
        if not question or not all_messages:
            return {'error': 'Нет данных для анализа'}
        
        # 1. Классификация
        topic, confidence = self.classifier.classify_question(question)
        
        # 2. Группировка сессий
        sessions = self.session_grouper.group_messages_by_sessions(all_messages, dialogue_id)
        
        # 3. Анализ релевантности
        relevance_scores = self.relevance_filter.get_relevance_scores(question, sessions)
        
        # 4. Ранжирование
        ranked_sessions = self.session_ranker.rank_sessions(question, sessions)
        
        # 5. Статистика
        stats = self.session_ranker.get_session_ranking_stats(question, sessions)
        
        return {
            'question': question,
            'topic': topic,
            'confidence': confidence,
            'strategy': 'topic_rag' if confidence >= self.config.classification_confidence_threshold else 'fallback',
            'total_sessions': len(sessions),
            'relevance_scores': relevance_scores,
            'ranked_sessions': ranked_sessions[:5],  # Топ-5
            'stats': stats,
            'config': {
                'confidence_threshold': self.config.classification_confidence_threshold,
                'max_sessions': self.config.max_relevant_sessions
            }
        }
    
    def update_config(self, new_config: Dict) -> None:
        """
        Обновляет конфигурацию RAG движка
        
        Args:
            new_config: Новые параметры конфигурации
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Обновляем конфигурацию в компонентах
        self.relevance_filter.config = self.config
        self.session_ranker.config = self.config
        self.session_ranker.scorer.config = self.config
    
    def get_engine_stats(self) -> Dict:
        """
        Получает статистику работы движка
        
        Returns:
            Словарь со статистикой
        """
        return {
            'config': {
                'confidence_threshold': self.config.classification_confidence_threshold,
                'max_relevant_sessions': self.config.max_relevant_sessions,
                'min_relevant_sessions': self.config.min_relevant_sessions
            },
            'available_topics': self.classifier.get_available_topics(),
            'session_grouper_stats': {
                'total_dialogues': len(self.session_grouper.grouped_sessions)
            }
        }


# Глобальный экземпляр для удобства использования
rag_engine = RAGEngine()


def process_question_with_rag(question: str, dialogue_id: str, 
                            all_messages: List[Message]) -> Tuple[str, Dict]:
    """
    Простая функция для обработки вопроса с RAG
    (для обратной совместимости)
    
    Args:
        question: Вопрос пользователя
        dialogue_id: ID диалога
        all_messages: Все сообщения диалога
        
    Returns:
        Tuple[промпт, метаданные]
    """
    return rag_engine.process_question(question, dialogue_id, all_messages)
