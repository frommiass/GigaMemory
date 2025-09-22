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
        
        # Группируем сообщения по сессиям
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
        
        # Группируем сообщения по сессиям
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
