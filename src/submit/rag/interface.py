"""
Интерфейс для интеграции RAG системы с основной моделью
"""
from typing import Dict, List, Tuple, Optional
from models import Message

from .engine import RAGEngine
from .config import DEFAULT_CONFIG


class RAGInterface:
    """Интерфейс для интеграции RAG системы"""
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.rag_engine = RAGEngine(config)
    
    def answer_question(self, question: str, dialogue_id: str, 
                       all_messages: List[Message]) -> str:
        """
        Отвечает на вопрос с использованием RAG системы
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Ответ на вопрос (промпт для модели)
        """
        if not question or not all_messages:
            return "У меня нет информации для ответа на этот вопрос."
        
        # Обрабатываем вопрос через RAG движок
        prompt, metadata = self.rag_engine.process_question(question, dialogue_id, all_messages)
        
        return prompt
    
    def get_question_context(self, question: str, dialogue_id: str, 
                           all_messages: List[Message]) -> Dict:
        """
        Получает контекст для вопроса без генерации ответа
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Словарь с контекстом и метаданными
        """
        if not question or not all_messages:
            return {'error': 'Нет данных для анализа'}
        
        # Получаем анализ вопроса
        analysis = self.rag_engine.get_question_analysis(question, dialogue_id, all_messages)
        
        return analysis
    
    def get_all_session_scores(self, question: str, dialogue_id: str, 
                              all_messages: List[Message]) -> Dict:
        """
        Получает оценки релевантности для ВСЕХ сессий
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Словарь с оценками всех сессий
        """
        if not question or not all_messages:
            return {'error': 'Нет данных для анализа'}
        
        # Группируем сообщения по сессиям (фильтрация уже применена в DataLoader)
        sessions = self.rag_engine.session_grouper.group_messages_by_sessions(all_messages, dialogue_id)
        
        # Классифицируем вопрос
        topic, confidence = self.rag_engine.classifier.classify_question(question)
        
        # Получаем оценки релевантности для всех сессий
        all_scores = self.rag_engine.relevance_filter.get_relevance_scores(question, sessions)
        
        # Получаем детальную информацию о каждой сессии
        session_info = self.rag_engine.relevance_filter.get_session_relevance_info(question, sessions)
        
        # Создаем полный словарь оценок
        full_scores = {}
        for session_id in sessions.keys():
            score = all_scores.get(session_id, 0.0)
            info = session_info.get(session_id, {})
            
            # Получаем сообщения сессии для анализа личных маркеров
            session_messages = sessions.get(session_id, [])
            
            # Анализируем личные маркеры
            from ..core.message_filter import calculate_personal_markers_score
            personal_analysis = calculate_personal_markers_score(session_messages)
            
            full_scores[session_id] = {
                'relevance_score': score,
                'is_relevant': score > 0.1,
                'status': 'релевантная' if score > 0.1 else 'низкая релевантность',
                'matched_keywords': info.get('matched_keywords', []),
                'content_length': info.get('content_length', 0),
                'message_count': info.get('message_count', 0),
                'personal_score': personal_analysis['personal_score'],
                'personal_markers_found': personal_analysis['personal_markers_found'],
                'personal_markers_per_message': personal_analysis['personal_markers_per_message'],
                'top_personal_markers': personal_analysis['top_personal_markers']
            }
        
        # Создаем список сессий, отсортированных по score (убывание) - с полной информацией
        sorted_sessions = sorted(
            full_scores.items(), 
            key=lambda x: x[1]['relevance_score'], 
            reverse=True
        )
        # Форматируем с порядковым номером, ID сессии, score и статусом
        sorted_sessions = [
            {
                'rank': i + 1,
                'session_id': int(session_id),
                'score': info['relevance_score'],
                'status': info['status'],
                'keywords': info['matched_keywords'],
                'personal_score': info['personal_score'],
                'personal_markers_found': info['personal_markers_found'],
                'personal_markers_per_message': info['personal_markers_per_message'],
                'top_personal_markers': info['top_personal_markers']
            }
            for i, (session_id, info) in enumerate(sorted_sessions)
        ]
        
        return {
            'question': question,
            'topic': topic,
            'confidence': confidence,
            'strategy': 'topic_rag' if confidence >= self.config.classification_confidence_threshold else 'fallback',
            'total_sessions': len(sessions),
            'relevant_sessions': len([s for s in full_scores.values() if s['is_relevant']]),
            'all_scores': full_scores,
            'sessions': sessions,
            'sorted_sessions': sorted_sessions
        }
    
    def classify_question(self, question: str) -> Tuple[Optional[str], float]:
        """
        Классифицирует вопрос по темам
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Tuple[тема, уверенность]
        """
        return self.rag_engine.classifier.classify_question(question)
    
    def get_relevant_sessions(self, question: str, dialogue_id: str, 
                            all_messages: List[Message]) -> Dict[str, List[Message]]:
        """
        Получает релевантные сессии для вопроса
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Словарь релевантных сессий
        """
        if not question or not all_messages:
            return {}
        
        # Группируем сообщения по сессиям (фильтрация уже применена в DataLoader)
        sessions = self.rag_engine.session_grouper.group_messages_by_sessions(all_messages, dialogue_id)
        
        # Находим релевантные сессии
        relevant_sessions = self.rag_engine.relevance_filter.find_relevant_sessions(
            question, sessions, self.config.max_relevant_sessions
        )
        
        return relevant_sessions
    
    def get_session_ranking(self, question: str, dialogue_id: str, 
                          all_messages: List[Message]) -> List[Tuple[str, float]]:
        """
        Получает ранжирование сессий по релевантности
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Список кортежей (session_id, score)
        """
        if not question or not all_messages:
            return []
        
        # Группируем сообщения по сессиям (фильтрация уже применена в DataLoader)
        sessions = self.rag_engine.session_grouper.group_messages_by_sessions(all_messages, dialogue_id)
        
        # Ранжируем сессии
        ranked_sessions = self.rag_engine.session_ranker.rank_sessions(
            question, sessions, self.config.max_relevant_sessions
        )
        
        return ranked_sessions
    
    def update_config(self, new_config: Dict) -> None:
        """
        Обновляет конфигурацию RAG системы
        
        Args:
            new_config: Новые параметры конфигурации
        """
        self.rag_engine.update_config(new_config)
        self.config = self.rag_engine.config
    
    def get_system_stats(self) -> Dict:
        """
        Получает статистику работы системы
        
        Returns:
            Словарь со статистикой
        """
        return self.rag_engine.get_engine_stats()
    
    def reset_dialogue_sessions(self, dialogue_id: str) -> None:
        """
        Сбрасывает сессии для конкретного диалога
        
        Args:
            dialogue_id: ID диалога
        """
        self.rag_engine.session_grouper.clear_dialogue_sessions(dialogue_id)
    
    def get_dialogue_sessions(self, dialogue_id: str) -> Dict[str, List[Message]]:
        """
        Получает все сессии диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь сессий диалога
        """
        return self.rag_engine.session_grouper.get_all_sessions(dialogue_id)
    
    def get_session_count(self, dialogue_id: str) -> int:
        """
        Получает количество сессий в диалоге
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Количество сессий
        """
        return self.rag_engine.session_grouper.get_session_count(dialogue_id)
    
    def validate_question(self, question: str) -> Dict[str, any]:
        """
        Валидирует вопрос и возвращает информацию о нем
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Словарь с информацией о вопросе
        """
        if not question or not question.strip():
            return {
                'valid': False,
                'error': 'Пустой вопрос',
                'topic': None,
                'confidence': 0.0
            }
        
        # Классифицируем вопрос
        topic, confidence = self.rag_engine.classifier.classify_question(question)
        
        return {
            'valid': True,
            'question': question.strip(),
            'topic': topic,
            'confidence': confidence,
            'strategy': 'topic_rag' if confidence >= self.config.classification_confidence_threshold else 'fallback',
            'length': len(question),
            'word_count': len(question.split())
        }


# Глобальный экземпляр для удобства использования
rag_interface = RAGInterface()


def answer_question_with_rag(question: str, dialogue_id: str, 
                           all_messages: List[Message]) -> str:
    """
    Простая функция для ответа на вопрос с RAG
    (для обратной совместимости)
    
    Args:
        question: Вопрос пользователя
        dialogue_id: ID диалога
        all_messages: Все сообщения диалога
        
    Returns:
        Ответ на вопрос
    """
    return rag_interface.answer_question(question, dialogue_id, all_messages)
