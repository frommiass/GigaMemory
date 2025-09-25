"""
Улучшенный RAG модуль с полной интеграцией через DI
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from ...core.interfaces import IRAGEngine, ProcessingResult
from .questions.classifier import QuestionClassifier
from .config import RAGConfig


@dataclass
class RAGContext:
    """Контекст для RAG обработки"""
    question: str
    dialogue_id: str
    topic: Optional[str] = None
    confidence: float = 0.0
    relevant_sessions: List[Dict] = None
    relevant_facts: List[Dict] = None
    compressed_context: Optional[str] = None
    memory_length: int = 0


class RAGModule(IRAGEngine):
    """
    Главный модуль RAG (Retrieval-Augmented Generation)
    Интегрирует все компоненты через Dependency Injection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Инициализация с конфигурацией"""
        self.config = RAGConfig(**config) if isinstance(config, dict) else config
        self.classifier = QuestionClassifier()
        self.logger = logging.getLogger(__name__)
        
        # Зависимости (будут установлены через DI)
        self.storage = None
        self.embeddings = None
        self.extractor = None
        self.compressor = None
        self.optimizer = None
        
        # Кэши
        self._session_cache = {}
        self._fact_cache = {}
    
    def set_dependencies(self, storage=None, embeddings=None, 
                        extractor=None, compressor=None, optimizer=None):
        """
        Устанавливает зависимости через Dependency Injection
        
        Args:
            storage: Модуль хранилища
            embeddings: Модуль эмбеддингов
            extractor: Модуль извлечения фактов
            compressor: Модуль сжатия
            optimizer: Модуль оптимизации
        """
        self.storage = storage
        self.embeddings = embeddings
        self.extractor = extractor
        self.compressor = compressor
        self.optimizer = optimizer
        self.logger.info("Dependencies set for RAG module")

    def process_question(self, question: str, dialogue_id: str) -> ProcessingResult:
        """
        Полная обработка вопроса с генерацией промпта
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            
        Returns:
            ProcessingResult с промптом и метаданными
        """
        try:
            # 1. Создаем контекст обработки
            context = RAGContext(
                question=question,
                dialogue_id=dialogue_id
            )
            
            # 2. Классифицируем вопрос
            context.topic, context.confidence = self._classify_question(question)
            
            # 3. Получаем релевантные сессии (если есть storage)
            if self.storage:
                context.relevant_sessions = self._get_relevant_sessions(
                    question, dialogue_id, context.topic, context.confidence
                )
            
            # 4. Получаем релевантные факты (если есть extractor)
            if self.extractor:
                context.relevant_facts = self._get_relevant_facts(
                    question, dialogue_id, context.topic
                )
            
            # 5. Гибридный поиск с эмбеддингами (если есть embeddings)
            if self.embeddings and context.relevant_sessions:
                context.relevant_sessions = self._rerank_with_embeddings(
                    question, context.relevant_sessions
                )
            
            # 6. Сжимаем контекст (если есть compressor)
            if self.compressor and (context.relevant_sessions or context.relevant_facts):
                context.compressed_context = self._compress_context(context)
            
            # 7. Генерируем оптимальный промпт
            prompt = self._generate_optimized_prompt(context)
            
            # 8. Оптимизируем промпт (если есть optimizer)
            if self.optimizer:
                prompt = self._optimize_prompt(prompt, context)
            
            return ProcessingResult(
                success=True,
                data=prompt,
                metadata=self._build_metadata(context)
            )
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return ProcessingResult(
                success=False,
                data=self._generate_fallback_prompt(question),
                metadata={'error': str(e)},
                error=str(e)
            )

    def _classify_question(self, question: str) -> Tuple[Optional[str], float]:
        """Классифицирует вопрос по темам"""
        return self.classifier.classify_question(question)

    def _get_relevant_sessions(self, question: str, dialogue_id: str, 
                              topic: Optional[str], confidence: float) -> List[Dict]:
        """Получает релевантные сессии из хранилища"""
        if not self.storage:
            return []
        
        try:
            # Используем тематический поиск если уверенность высокая
            if confidence >= self.config.classification_confidence_threshold and topic:
                result = self.storage.find_by_topic(dialogue_id, topic)
            else:
                # Иначе используем поиск по ключевым словам
                result = self.storage.search_sessions(dialogue_id, question)
            
            if result.success:
                sessions = result.data
                # Ограничиваем количество сессий
                return sessions[:self.config.max_relevant_sessions]
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting sessions: {e}")
            return []

    def _get_relevant_facts(self, question: str, dialogue_id: str, 
                           topic: Optional[str]) -> List[Dict]:
        """Получает релевантные факты через extractor"""
        if not self.extractor:
            return []
        
        try:
            # Получаем факты для диалога
            result = self.extractor.get_facts(dialogue_id)
            if not result.success:
                return []
            
            all_facts = result.data
            
            # Фильтруем по теме если она определена
            if topic:
                relevant_facts = [
                    fact for fact in all_facts
                    if fact.get('topic') == topic or topic in fact.get('content', '')
                ]
            else:
                # Простой поиск по ключевым словам
                question_words = set(question.lower().split())
                relevant_facts = []
                
                for fact in all_facts:
                    fact_words = set(fact.get('content', '').lower().split())
                    if question_words.intersection(fact_words):
                        relevant_facts.append(fact)
            
            # Ограничиваем количество фактов
            return relevant_facts[:10]
            
        except Exception as e:
            self.logger.error(f"Error getting facts: {e}")
            return []

    def _rerank_with_embeddings(self, question: str, sessions: List[Dict]) -> List[Dict]:
        """Переранжирует сессии используя векторный поиск"""
        if not self.embeddings or not sessions:
            return sessions
        
        try:
            # Получаем эмбеддинг вопроса
            query_result = self.embeddings.encode(question)
            if not query_result.success:
                return sessions
            
            query_embedding = query_result.data
            
            # Вычисляем схожесть для каждой сессии
            scored_sessions = []
            for session in sessions:
                session_text = session.get('content', '')
                if not session_text:
                    continue
                
                # Получаем эмбеддинг сессии
                session_result = self.embeddings.encode(session_text)
                if session_result.success:
                    session_embedding = session_result.data
                    
                    # Вычисляем косинусную схожесть
                    similarity = self._cosine_similarity(query_embedding, session_embedding)
                    session['relevance_score'] = similarity
                    scored_sessions.append(session)
            
            # Сортируем по убыванию схожести
            scored_sessions.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            return scored_sessions
            
        except Exception as e:
            self.logger.error(f"Error reranking with embeddings: {e}")
            return sessions

    def _compress_context(self, context: RAGContext) -> str:
        """Сжимает контекст для оптимизации промпта"""
        if not self.compressor:
            return ""
        
        try:
            # Объединяем весь контекст
            full_context = []
            
            # Добавляем сессии
            if context.relevant_sessions:
                for session in context.relevant_sessions:
                    content = session.get('content', '')
                    if content:
                        full_context.append(f"[Сессия] {content}")
            
            # Добавляем факты
            if context.relevant_facts:
                for fact in context.relevant_facts:
                    content = fact.get('content', '')
                    if content:
                        full_context.append(f"[Факт] {content}")
            
            if not full_context:
                return ""
            
            # Сжимаем объединенный контекст
            combined_text = "\n\n".join(full_context)
            
            # Проверяем длину
            if len(combined_text) <= 2000:
                return combined_text  # Не нужно сжимать
            
            # Сжимаем через compressor
            result = self.compressor.compress(combined_text, target_length=2000)
            if result.success:
                return result.data
            
            # Fallback - обрезаем до 2000 символов
            return combined_text[:2000]
            
        except Exception as e:
            self.logger.error(f"Error compressing context: {e}")
            return ""

    def _generate_optimized_prompt(self, context: RAGContext) -> str:
        """
        Генерирует оптимизированный промпт на основе контекста
        Использует разные шаблоны для разных типов вопросов
        """
        # Выбираем шаблон на основе темы и уверенности
        if context.confidence >= 0.8 and context.topic:
            template = self._get_topic_template(context.topic)
        elif context.confidence >= 0.5:
            template = self._get_mixed_template()
        else:
            template = self._get_fallback_template()
        
        # Подготавливаем контекстную информацию
        memory_context = ""
        
        # Используем сжатый контекст если есть
        if context.compressed_context:
            memory_context = context.compressed_context
        else:
            # Иначе формируем из сессий и фактов
            parts = []
            
            if context.relevant_sessions:
                for i, session in enumerate(context.relevant_sessions[:3], 1):
                    content = session.get('content', '')
                    if content:
                        parts.append(f"[Контекст {i}]\n{content[:500]}")
            
            if context.relevant_facts:
                parts.append("\n[Известные факты]")
                for fact in context.relevant_facts[:5]:
                    content = fact.get('content', '')
                    if content:
                        parts.append(f"• {content}")
            
            memory_context = "\n\n".join(parts)
        
        # Ограничиваем контекст до 2000 символов
        if len(memory_context) > 2000:
            memory_context = memory_context[:1997] + "..."
        
        # Заполняем шаблон
        prompt = template.format(
            topic=context.topic or "общая",
            confidence=context.confidence,
            question=context.question,
            context=memory_context or "Контекст не найден",
            dialogue_id=context.dialogue_id
        )
        
        return prompt

    def _get_topic_template(self, topic: str) -> str:
        """Возвращает шаблон для конкретной темы"""
        templates = {
            "sports": """Тема: Спорт и фитнес (уверенность: {confidence:.1%})
Контекст из предыдущих обсуждений:
{context}
Вопрос пользователя: {question}
Дай подробный ответ, используя информацию из контекста. Если в контексте есть личный опыт пользователя, обязательно учти его.""",
            "work": """Тема: Работа и карьера (уверенность: {confidence:.1%})
Рабочий контекст:
{context}
Вопрос: {question}
Ответь профессионально, учитывая специфику рабочих вопросов из контекста.""",
            "family": """Тема: Семья и отношения (уверенность: {confidence:.1%})
Семейный контекст:
{context}
Вопрос: {question}
Дай чуткий и внимательный ответ, учитывая семейную ситуацию из контекста."""
        }
        # Возвращаем специфичный шаблон или общий
        return templates.get(topic, self._get_default_template())

    def _get_mixed_template(self) -> str:
        """Шаблон для средней уверенности"""
        return """Возможная тема: {topic} (уверенность: {confidence:.1%})
Найденная информация:
{context}
Вопрос: {question}
Ответь на вопрос, используя доступный контекст. Если информации недостаточно, укажи это."""

    def _get_fallback_template(self) -> str:
        """Шаблон для низкой уверенности"""
        return """Вопрос пользователя: {question}
Доступный контекст:
{context}
Постарайся ответить на вопрос, используя любую релевантную информацию из контекста."""

    def _get_default_template(self) -> str:
        """Стандартный шаблон"""
        return """Тема: {topic}
Уверенность: {confidence:.1%}
Контекст:
{context}
Вопрос: {question}
Пожалуйста, ответь на вопрос, основываясь на предоставленном контексте."""

    def _optimize_prompt(self, prompt: str, context: RAGContext) -> str:
        """Оптимизирует промпт через optimizer если доступен"""
        if not self.optimizer:
            return prompt
        
        try:
            result = self.optimizer.optimize(prompt)
            if result.success:
                return result.data
        except Exception as e:
            self.logger.error(f"Error optimizing prompt: {e}")
        
        return prompt

    def _generate_fallback_prompt(self, question: str) -> str:
        """Генерирует fallback промпт при ошибках"""
        return f"""Вопрос пользователя: {question}
Пожалуйста, ответь на вопрос настолько полно, насколько это возможно."""

    def _build_metadata(self, context: RAGContext) -> Dict[str, Any]:
        """Собирает метаданные обработки"""
        return {
            'question': context.question,
            'dialogue_id': context.dialogue_id,
            'topic': context.topic,
            'confidence': context.confidence,
            'strategy': self._determine_strategy(context),
            'sessions_found': len(context.relevant_sessions) if context.relevant_sessions else 0,
            'facts_found': len(context.relevant_facts) if context.relevant_facts else 0,
            'context_compressed': context.compressed_context is not None,
            'memory_length': len(context.compressed_context) if context.compressed_context else 0,
            'timestamp': datetime.now().isoformat()
        }

    def _determine_strategy(self, context: RAGContext) -> str:
        """Определяет использованную стратегию"""
        if context.confidence >= self.config.classification_confidence_threshold:
            if context.relevant_facts and context.relevant_sessions:
                return "hybrid_topic_rag"
            elif context.relevant_facts:
                return "fact_based_rag"
            elif context.relevant_sessions:
                return "session_based_rag"
            else:
                return "topic_only"
        else:
            return "fallback"

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисляет косинусную схожесть между векторами"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    # === Дополнительные методы интерфейса ===

    def find_relevant_sessions(self, question: str, dialogue_id: str) -> ProcessingResult:
        """Находит релевантные сессии для вопроса"""
        try:
            topic, confidence = self._classify_question(question)
            sessions = self._get_relevant_sessions(question, dialogue_id, topic, confidence)
            
            return ProcessingResult(
                success=True,
                data=sessions,
                metadata={
                    'topic': topic,
                    'confidence': confidence,
                    'found': len(sessions)
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                error=str(e)
            )

    def generate_answer(self, question: str, context: str) -> ProcessingResult:
        """Генерирует ответ на основе контекста"""
        try:
            prompt = f"""На основе следующего контекста ответь на вопрос.
Контекст:
{context[:2000]}
Вопрос: {question}
Дай точный и полезный ответ:"""
            return ProcessingResult(
                success=True,
                data=prompt,
                metadata={'context_length': len(context)}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data="",
                error=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы модуля"""
        return {
            'config': {
                'confidence_threshold': self.config.classification_confidence_threshold,
                'max_relevant_sessions': self.config.max_relevant_sessions
            },
            'dependencies': {
                'storage': self.storage is not None,
                'embeddings': self.embeddings is not None,
                'extractor': self.extractor is not None,
                'compressor': self.compressor is not None,
                'optimizer': self.optimizer is not None
            },
            'cache_stats': {
                'sessions_cached': len(self._session_cache),
                'facts_cached': len(self._fact_cache)
            },
            'available_topics': self.classifier.get_available_topics()
        }

    def clear_cache(self):
        """Очищает внутренние кэши"""
        self._session_cache.clear()
        self._fact_cache.clear()
        self.logger.info("RAG caches cleared")