"""
RAG движок с векторным поиском
"""
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import numpy as np

from models import Message
from .engine import RAGEngine
from .config import RAGConfig
from ..embeddings.embedding_engine import EmbeddingEngine, EmbeddingConfig
from ..embeddings.vector_store import VectorStore, OptimizedVectorStore, SearchResult
from ..filters.session_grouper import extract_session_content
from ..questions.classifier import QuestionClassifier
from ..prompts.topic_prompts import get_topic_prompt
from ..prompts.fallback_prompts import get_fallback_prompt
from ..extraction.fact_extractor import RuleBasedFactExtractor
from ..extraction.fact_database import FactDatabase

logger = logging.getLogger(__name__)


class VectorRAGConfig(RAGConfig):
    """Расширенная конфигурация для векторного RAG"""
    
    # Настройки эмбеддингов
    embedding_model: str = "cointegrated/rubert-tiny2"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    embedding_cache_dir: str = ".embedding_cache"
    
    # Настройки векторного поиска
    vector_search_metric: str = "cosine"
    vector_search_top_k: int = 5
    vector_search_threshold: float = 0.7
    
    # Настройки гибридного поиска
    use_hybrid_search: bool = True
    keyword_weight: float = 0.3
    vector_weight: float = 0.7


class VectorRAGEngine(RAGEngine):
    """RAG движок с векторным поиском"""
    
    def __init__(self, config: Optional[VectorRAGConfig] = None):
        """
        Инициализация векторного RAG движка
        
        Args:
            config: Конфигурация движка
        """
        # Используем расширенную конфигурацию
        self.config = config or VectorRAGConfig()
        
        # Инициализируем базовый движок
        super().__init__(self.config)
        
        # Создаем движок эмбеддингов
        embedding_config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            batch_size=self.config.embedding_batch_size,
            cache_dir=self.config.embedding_cache_dir
        )
        self.embedding_engine = EmbeddingEngine(embedding_config)
        
        # Векторные хранилища для каждого диалога
        self.vector_stores: Dict[str, VectorStore] = {}
        
        # Кэш индексов
        self.indexed_dialogues: set = set()

        # Извлечение фактов
        self.fact_extractor = RuleBasedFactExtractor()
        self.fact_database = FactDatabase(conflict_strategy="latest")
        
        logger.info(f"Инициализирован VectorRAGEngine с моделью {self.config.embedding_model}")
    
    def index_dialogue(self, dialogue_id: str, sessions: Dict[str, List[Message]],
                      force_reindex: bool = False) -> Dict[str, Any]:
        """
        Индексирует все сессии диалога для векторного поиска
        
        Args:
            dialogue_id: ID диалога
            sessions: Словарь сессий {session_id: [messages]}
            force_reindex: Принудительная переиндексация
            
        Returns:
            Статистика индексации
        """
        # Проверяем, нужна ли индексация
        if dialogue_id in self.indexed_dialogues and not force_reindex:
            logger.debug(f"Диалог {dialogue_id} уже проиндексирован")
            return {'status': 'already_indexed', 'sessions': 0}
        
        logger.info(f"Индексация диалога {dialogue_id}: {len(sessions)} сессий")
        
        # Создаем новое векторное хранилище
        store = OptimizedVectorStore(
            metric=self.config.vector_search_metric,
            normalize=True
        )
        
        # Подготавливаем данные для индексации
        texts_to_index = []
        metadata_list = []
        session_ids = []
        
        all_facts = []
        for session_id, messages in sessions.items():
            # Извлекаем текст сессии (только сообщения пользователя)
            session_text = extract_session_content(messages)
            
            if not session_text.strip():
                continue
            
            # Подготавливаем метаданные
            metadata = {
                'session_id': session_id,
                'dialogue_id': dialogue_id,
                'message_count': len(messages),
                'text_length': len(session_text)
            }
            
            # Классифицируем содержимое сессии
            topic, confidence = self.classifier.classify_question(session_text[:200])
            if topic:
                metadata['topic'] = topic
                metadata['topic_confidence'] = confidence
            
            texts_to_index.append(session_text)
            metadata_list.append(metadata)
            session_ids.append(session_id)

            # Извлекаем факты из сессии
            try:
                facts = self.fact_extractor.extract_facts_from_text(session_text, session_id, dialogue_id)
                if facts:
                    all_facts.extend(facts)
            except Exception as e:
                logger.warning(f"Ошибка извлечения фактов для сессии {session_id}: {e}")
        
        # Создаем эмбеддинги батчем
        if texts_to_index:
            logger.debug(f"Создаем эмбеддинги для {len(texts_to_index)} сессий")
            embeddings = self.embedding_engine.encode(texts_to_index, show_progress=False)
            
            # Добавляем в векторное хранилище
            for session_id, text, embedding, metadata in zip(session_ids, texts_to_index, 
                                                            embeddings, metadata_list):
                store.add(
                    doc_id=f"{dialogue_id}_{session_id}",
                    vector=embedding,
                    metadata=metadata,
                    text=text
                )
        
        # Сохраняем хранилище
        self.vector_stores[dialogue_id] = store
        self.indexed_dialogues.add(dialogue_id)

        # Сохраняем и индексируем факты
        if all_facts:
            self.fact_database.add_facts(dialogue_id, all_facts)
            self._index_facts(dialogue_id, all_facts)
        
        stats = {
            'status': 'indexed',
            'dialogue_id': dialogue_id,
            'sessions_indexed': len(texts_to_index),
            'total_sessions': len(sessions),
            'vector_store_size': store.size(),
            'embedding_dim': self.embedding_engine.get_embedding_dim(),
            'facts_extracted': len(all_facts),
            'conflicts_resolved': self.fact_database.stats.conflicts_resolved if hasattr(self.fact_database, 'stats') else 0
        }
        
        logger.info(f"Индексация завершена: {stats}")
        return stats

    def _index_facts(self, dialogue_id: str, facts) -> None:
        """Индексирует факты как отдельные документы в векторном хранилище диалога."""
        if dialogue_id not in self.vector_stores:
            return
        store = self.vector_stores[dialogue_id]
        try:
            # Текстовые представления фактов и батч-энкодинг
            fact_texts = []
            fact_metas = []
            fact_ids = []
            for fact in facts:
                # Ожидаем, что у факта есть поля type, to_dict()/to_natural_text(), confidence, session_id
                fact_type = getattr(fact.type, 'value', str(getattr(fact, 'type', 'fact')))
                if hasattr(fact, 'to_natural_text'):
                    fact_text = f"{fact_type}: {fact.to_natural_text()}"
                else:
                    fact_text = f"{fact_type}: {str(getattr(fact, 'object', ''))}"
                fact_texts.append(fact_text)
                fact_metas.append({
                    'type': 'fact',
                    'fact_type': fact_type,
                    'confidence': getattr(getattr(fact, 'confidence', None), 'score', 0.0),
                    'session_id': getattr(fact, 'session_id', ''),
                    'dialogue_id': getattr(fact, 'dialogue_id', dialogue_id)
                })
                fact_ids.append(f"fact_{getattr(fact, 'id', '')}")

            if not fact_texts:
                return

            embeddings = self.embedding_engine.encode(fact_texts, show_progress=False)
            for doc_id, embedding, meta, text in zip(fact_ids, embeddings, fact_metas, fact_texts):
                store.add(doc_id=doc_id, vector=embedding, metadata=meta, text=text)
        except Exception as e:
            logger.warning(f"Не удалось проиндексировать факты для диалога {dialogue_id}: {e}")
    
    def find_relevant_sessions_vector(self, question: str, dialogue_id: str,
                                    k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        Находит релевантные сессии через векторный поиск
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            k: Количество результатов
            
        Returns:
            Список кортежей (session_id, score, metadata)
        """
        k = k or self.config.vector_search_top_k
        
        # Проверяем, проиндексирован ли диалог
        if dialogue_id not in self.vector_stores:
            logger.warning(f"Диалог {dialogue_id} не проиндексирован")
            return []
        
        store = self.vector_stores[dialogue_id]
        
        # Создаем эмбеддинг вопроса
        question_embedding = self.embedding_engine.encode(question)
        
        # Выполняем векторный поиск
        results = store.search(
            query_vector=question_embedding,
            k=k,
            threshold=self.config.vector_search_threshold
        )
        
        # Преобразуем результаты
        relevant_sessions = []
        for result in results:
            session_id = result.metadata.get('session_id', '')
            score = result.score
            metadata = result.metadata
            relevant_sessions.append((session_id, score, metadata))
        
        logger.debug(f"Найдено {len(relevant_sessions)} релевантных сессий для вопроса")
        return relevant_sessions
    
    def process_question(self, question: str, dialogue_id: str,
                        all_messages: List[Message]) -> Tuple[str, Dict]:
        """
        Обрабатывает вопрос с использованием векторного поиска
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            all_messages: Все сообщения диалога
            
        Returns:
            Tuple[промпт, метаданные]
        """
        if not question or not all_messages:
            return "У меня нет информации для ответа на этот вопрос.", {}
        
        # 1. Группируем сообщения по сессиям
        sessions = self.session_grouper.group_messages_by_sessions(all_messages, dialogue_id)
        
        # 2. Индексируем диалог если еще не проиндексирован
        if dialogue_id not in self.indexed_dialogues:
            index_stats = self.index_dialogue(dialogue_id, sessions)
            logger.info(f"Проиндексировано: {index_stats}")
        
        # 3. Классифицируем вопрос
        topic, confidence = self.classifier.classify_question(question)
        
        # 4. Выполняем векторный поиск
        vector_results = self.find_relevant_sessions_vector(question, dialogue_id)
        
        # 5. Если включен гибридный поиск, комбинируем с keyword search
        if self.config.use_hybrid_search:
            relevant_sessions = self._hybrid_search(
                question, sessions, vector_results, topic, confidence
            )
        else:
            # Используем только векторный поиск
            relevant_sessions = self._convert_vector_results_to_sessions(
                vector_results, sessions
            )
        
        # 6. Формируем контекст с учетом фактов
        memory_text = self._build_memory_context(relevant_sessions)
        facts_text, relevant_facts = self._build_facts_context(question, dialogue_id)
        if facts_text:
            memory_text = f"ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:\n{facts_text}\n\nКОНТЕКСТ ДИАЛОГОВ:\n" + memory_text
        
        # 7. Генерируем промпт
        if topic and confidence >= self.config.classification_confidence_threshold:
            prompt = get_topic_prompt(topic, question, memory_text)
            strategy = 'vector_topic_rag'
        else:
            prompt = get_fallback_prompt(question, memory_text)
            strategy = 'vector_fallback'
        
        # 8. Собираем метаданные
        metadata = {
            'strategy': strategy,
            'topic': topic,
            'confidence': confidence,
            'total_sessions': len(sessions),
            'vector_search_results': len(vector_results),
            'selected_sessions': len(relevant_sessions),
            'selected_session_ids': list(relevant_sessions.keys()),
            'memory_length': len(memory_text),
            'vector_scores': [(r[0], r[1]) for r in vector_results[:5]],  # Топ-5 scores
            'facts_used': len(relevant_facts) if isinstance(relevant_facts, list) else 0
        }
        
        return prompt, metadata

    def _build_facts_context(self, question: str, dialogue_id: str) -> Tuple[str, List[Any]]:
        """Ищет релевантные факты и возвращает текстовый блок и список фактов."""
        try:
            if not hasattr(self, 'fact_database') or self.fact_database is None:
                return "", []
            # Ищем по текстовому запросу, берём топ по уверенности
            facts = self.fact_database.query_facts(dialogue_id, query=question, min_confidence=0.5)
            if not facts:
                # Фолбэк: возьмём самые уверенные общие факты
                facts = sorted(self.fact_database.get_facts(dialogue_id), key=lambda f: f.confidence.score, reverse=True)[:5]
            if not facts:
                return "", []
            lines = []
            for fact in facts[:10]:
                fact_type = getattr(fact.type, 'value', 'fact')
                if hasattr(fact, 'to_natural_text'):
                    text = fact.to_natural_text()
                else:
                    text = str(getattr(fact, 'object', ''))
                conf = getattr(getattr(fact, 'confidence', None), 'score', 0.0)
                lines.append(f"- [{conf:.2f}] {fact_type}: {text}")
            return "\n".join(lines), facts
        except Exception:
            return "", []
    
    def _hybrid_search(self, question: str, sessions: Dict[str, List[Message]],
                      vector_results: List[Tuple[str, float, Dict]],
                      topic: Optional[str], confidence: float) -> Dict[str, List[Message]]:
        """
        Выполняет гибридный поиск, комбинируя векторный и keyword поиск
        
        Args:
            question: Вопрос пользователя
            sessions: Все сессии диалога
            vector_results: Результаты векторного поиска
            topic: Тема вопроса
            confidence: Уверенность в классификации
            
        Returns:
            Словарь релевантных сессий
        """
        # Получаем результаты keyword search через базовый движок
        if topic and confidence >= self.config.classification_confidence_threshold:
            keyword_sessions = self.relevance_filter.find_relevant_sessions(
                question, sessions, self.config.max_relevant_sessions
            )
        else:
            keyword_sessions = {}
        
        # Объединяем результаты с весами
        combined_scores = {}
        
        # Добавляем scores от векторного поиска
        for session_id, score, metadata in vector_results:
            if session_id in sessions:
                combined_scores[session_id] = score * self.config.vector_weight
        
        # Добавляем scores от keyword search
        for session_id in keyword_sessions:
            if session_id in combined_scores:
                # Сессия найдена обоими методами - усиливаем score
                combined_scores[session_id] += self.config.keyword_weight
            else:
                # Сессия найдена только keyword search
                combined_scores[session_id] = self.config.keyword_weight * 0.5
        
        # Сортируем по combined score и берем топ
        sorted_sessions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_sessions = sorted_sessions[:self.config.max_relevant_sessions]
        
        # Возвращаем сессии
        result = {}
        for session_id, score in top_sessions:
            if session_id in sessions:
                result[session_id] = sessions[session_id]
        
        logger.debug(f"Гибридный поиск: {len(result)} сессий из {len(vector_results)} векторных и {len(keyword_sessions)} keyword")
        return result
    
    def _convert_vector_results_to_sessions(self, vector_results: List[Tuple[str, float, Dict]],
                                          sessions: Dict[str, List[Message]]) -> Dict[str, List[Message]]:
        """
        Преобразует результаты векторного поиска в словарь сессий
        
        Args:
            vector_results: Результаты векторного поиска
            sessions: Все сессии диалога
            
        Returns:
            Словарь релевантных сессий
        """
        result = {}
        for session_id, score, metadata in vector_results[:self.config.max_relevant_sessions]:
            if session_id in sessions:
                result[session_id] = sessions[session_id]
        return result
    
    def get_vector_search_analysis(self, question: str, dialogue_id: str) -> Dict[str, Any]:
        """
        Получает детальный анализ векторного поиска
        
        Args:
            question: Вопрос пользователя
            dialogue_id: ID диалога
            
        Returns:
            Словарь с анализом
        """
        if dialogue_id not in self.vector_stores:
            return {'error': 'Диалог не проиндексирован'}
        
        store = self.vector_stores[dialogue_id]
        
        # Создаем эмбеддинг вопроса
        question_embedding = self.embedding_engine.encode(question)
        
        # Выполняем поиск с разными k
        results_5 = store.search(question_embedding, k=5)
        results_10 = store.search(question_embedding, k=10)
        
        # Анализируем распределение scores
        scores = [r.score for r in results_10]
        
        analysis = {
            'question': question,
            'dialogue_id': dialogue_id,
            'vector_store_size': store.size(),
            'embedding_dim': self.embedding_engine.get_embedding_dim(),
            'top_5_results': [
                {
                    'session_id': r.metadata.get('session_id'),
                    'score': r.score,
                    'text_preview': r.text[:100] if r.text else None,
                    'metadata': r.metadata
                }
                for r in results_5
            ],
            'score_distribution': {
                'max': max(scores) if scores else 0,
                'min': min(scores) if scores else 0,
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0
            },
            'high_relevance_count': sum(1 for s in scores if s >= self.config.vector_search_threshold)
        }
        
        return analysis
    
    def save_indices(self, save_dir: str):
        """
        Сохраняет все векторные индексы на диск
        
        Args:
            save_dir: Директория для сохранения
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        for dialogue_id, store in self.vector_stores.items():
            filepath = save_path / f"vector_store_{dialogue_id}.pkl"
            store.save(str(filepath))
        
        # Сохраняем кэш эмбеддингов
        cache_file = save_path / "embedding_cache.pkl"
        self.embedding_engine.save_cache(str(cache_file))
        
        logger.info(f"Сохранено {len(self.vector_stores)} индексов в {save_dir}")
    
    def load_indices(self, save_dir: str):
        """
        Загружает векторные индексы с диска
        
        Args:
            save_dir: Директория с сохраненными индексами
        """
        save_path = Path(save_dir)
        if not save_path.exists():
            logger.warning(f"Директория {save_dir} не существует")
            return
        
        # Загружаем кэш эмбеддингов
        cache_file = save_path / "embedding_cache.pkl"
        if cache_file.exists():
            self.embedding_engine.load_cache(str(cache_file))
        
        # Загружаем векторные хранилища
        for filepath in save_path.glob("vector_store_*.pkl"):
            dialogue_id = filepath.stem.replace("vector_store_", "")
            store = OptimizedVectorStore()
            store.load(str(filepath))
            self.vector_stores[dialogue_id] = store
            self.indexed_dialogues.add(dialogue_id)
        
        logger.info(f"Загружено {len(self.vector_stores)} индексов из {save_dir}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику движка"""
        stats = super().get_engine_stats()
        
        # Добавляем статистику векторного поиска
        stats['vector_search'] = {
            'indexed_dialogues': len(self.indexed_dialogues),
            'total_vectors': sum(store.size() for store in self.vector_stores.values()),
            'embedding_model': self.config.embedding_model,
            'embedding_dim': self.embedding_engine.get_embedding_dim(),
            'cache_size': len(self.embedding_engine.cache) if self.embedding_engine.cache else 0
        }
        
        return stats