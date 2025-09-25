# modules/rag/module.py

from typing import Dict, Any
from ...core.interfaces import IRAGEngine, ProcessingResult

from .compressed_rag_engine import CompressedRAGEngine, CompressedRAGConfig
from .questions.classifier import QuestionClassifier


class RAGModule(IRAGEngine):
    """Модуль RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Создаем конфигурацию RAG
        rag_config = CompressedRAGConfig(
            max_relevant_sessions=config.get('top_k', 5),
            enable_compression=config.get('use_compression', True),
            use_hierarchical=config.get('use_hierarchical', False)
        )
        
        # Инициализируем движок
        self.engine = CompressedRAGEngine(rag_config)
        self.classifier = QuestionClassifier()
    
    def process_question(self, question: str, dialogue_id: str) -> ProcessingResult:
        """Обрабатывает вопрос и генерирует промпт"""
        try:
            # Классифицируем вопрос
            topic, confidence = self.classifier.classify_question(question)
            
            # Обрабатываем через движок
            # Здесь нужна интеграция с другими модулями через DI
            prompt = self._generate_prompt(question, dialogue_id, topic, confidence)
            
            return ProcessingResult(
                success=True,
                data=prompt,
                metadata={
                    'topic': topic,
                    'confidence': confidence,
                    'dialogue_id': dialogue_id
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data="",
                metadata={},
                error=str(e)
            )
    
    def find_relevant_sessions(self, question: str, dialogue_id: str) -> ProcessingResult:
        """Находит релевантные сессии"""
        try:
            # Здесь должна быть логика поиска релевантных сессий
            # Пока заглушка
            relevant_sessions = []
            
            return ProcessingResult(
                success=True,
                data=relevant_sessions,
                metadata={'found': len(relevant_sessions)}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )
    
    def generate_answer(self, question: str, context: str) -> ProcessingResult:
        """Генерирует ответ на основе контекста"""
        try:
            # Формируем промпт для ответа
            prompt = f"""На основе следующего контекста ответь на вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:"""
            
            return ProcessingResult(
                success=True,
                data=prompt,
                metadata={'context_length': len(context)}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data="",
                metadata={},
                error=str(e)
            )
    
    def _generate_prompt(self, question: str, dialogue_id: str, 
                        topic: str, confidence: float) -> str:
        """Внутренний метод генерации промпта"""
        # Упрощенная версия
        return f"""Тема: {topic} (уверенность: {confidence:.2f})
Вопрос: {question}

Используя информацию из диалога {dialogue_id}, ответь на вопрос пользователя."""