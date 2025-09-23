"""
Оптимизированная версия SmartMemory с полной интеграцией всех компонентов
"""
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import threading
import time

from models import Message
from .llm_inference import ModelInference
from .rag.compressed_rag_engine import CompressedRAGEngine, CompressedRAGConfig
from .extraction import FactDatabase, SmartFactExtractor
from .optimization.cache_manager import CacheManager, BatchProcessor
from .config_loader import config_manager

logger = logging.getLogger(__name__)


class OptimizedSmartMemory:
    """Финальная оптимизированная версия интеллектуальной памяти"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация оптимизированной системы
        
        Args:
            model_path: Путь к модели (если None, берется из конфига)
        """
        # Загружаем конфигурацию
        self.system_config = config_manager.get_config()
        model_path = model_path or self.system_config.model_path
        
        # Конвертируем в SmartMemoryConfig
        self.config = self.system_config.to_smart_memory_config()
        
        # Инициализируем модель
        self.model = ModelInference(model_path)
        
        # Создаем оптимизированный RAG движок
        rag_config = self._create_rag_config()
        self.rag_engine = CompressedRAGEngine(rag_config)
        
        # Создаем менеджер кэша
        cache_config = self.system_config.cache_config
        self.cache_manager = CacheManager(
            max_size=cache_config.get('max_size', 10000),
            max_memory_mb=cache_config.get('max_memory_mb', 1024),
            eviction_strategy=cache_config.get('eviction_strategy', 'lru'),
            ttl_seconds=cache_config.get('default_ttl', 3600)
        )
        
        # Создаем батч-процессор
        batch_config = self.system_config.batch_config
        self.batch_processor = BatchProcessor(
            batch_size=batch_config.get('batch_size', 32),
            max_wait_time=batch_config.get('max_wait_time', 1.0)
        )
        
        # Извлекатель фактов с кэшированием
        if self.config.use_fact_extraction:
            self.fact_extractor = SmartFactExtractor(self.model, use_rules_first=True)
            self.fact_database = FactDatabase(
                conflict_strategy=self.system_config.fact_extraction_config.get('conflict_strategy', 'latest')
            )
        
        # Пул потоков для параллельной обработки
        num_workers = batch_config.get('num_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Фоновая обработка
        self.background_thread = None
        if batch_config.get('enabled', True):
            self._start_background_processing()
        
        # Автосохранение
        self.autosave_thread = None
        persistence_config = self.system_config.persistence_config
        if persistence_config.get('autosave', True):
            self._start_autosave()
        
        # Статистика и метрики
        self.metrics = {
            'total_queries': 0,
            'avg_query_time': 0.0,
            'cache_hit_rate': 0.0,
            'compression_ratio': 0.0,
            'facts_per_dialogue': 0.0
        }
        
        logger.info("OptimizedSmartMemory инициализирована")
    
    def _create_rag_config(self) -> CompressedRAGConfig:
        """Создает конфигурацию RAG из системной конфигурации"""
        from .compression import CompressionLevel, CompressionMethod
        
        config = CompressedRAGConfig()
        
        # Embedding настройки
        emb_config = self.system_config.embedding_config
        config.embedding_model = emb_config.get('model_name')
        config.embedding_device = emb_config.get('device', 'cuda')
        config.embedding_batch_size = emb_config.get('batch_size', 32)
        
        # Vector store настройки
        vs_config = self.system_config.vector_store_config
        config.vector_search_metric = vs_config.get('metric', 'cosine')
        config.vector_search_top_k = vs_config.get('top_k', 5)
        config.vector_search_threshold = vs_config.get('threshold', 0.7)
        
        # Compression настройки
        comp_config = self.system_config.compression_config
        config.enable_compression = comp_config.get('enabled', True)
        config.compression_level = CompressionLevel[comp_config.get('level', 'moderate').upper()]
        config.compression_method = CompressionMethod[comp_config.get('method', 'hybrid').upper()]
        config.compression_target_ratio = comp_config.get('target_ratio', 0.3)
        config.use_hierarchical = comp_config.get('hierarchical', True)
        
        # RAG настройки
        rag_config = self.system_config.rag_config
        config.use_hybrid_search = rag_config.get('use_hybrid_search', True)
        config.keyword_weight = rag_config.get('keyword_weight', 0.3)
        config.vector_weight = rag_config.get('vector_weight', 0.7)
        
        return config
    
    def process_dialogue_optimized(self, dialogue_id: str, messages: List[Message]) -> Dict[str, Any]:
        """
        Оптимизированная обработка диалога с кэшированием и батчингом
        
        Args:
            dialogue_id: ID диалога
            messages: Сообщения диалога
            
        Returns:
            Статистика обработки
        """
        start_time = time.time()
        
        # Проверяем кэш
        cache_key = f"dialogue_{dialogue_id}_{len(messages)}"
        cached_stats = self.cache_manager.get(cache_key, cache_type="general")
        if cached_stats:
            logger.info(f"Диалог {dialogue_id} найден в кэше")
            return cached_stats
        
        stats = {
            'dialogue_id': dialogue_id,
            'messages_count': len(messages),
            'cache_used': False
        }
        
        # 1. Группируем по сессиям
        from .filters.session_grouper import SessionGrouper
        grouper = SessionGrouper()
        sessions = grouper.group_messages_by_sessions(messages, dialogue_id)
        stats['sessions_count'] = len(sessions)
        
        # 2. Параллельная обработка сессий
        futures = []
        
        for session_id, session_messages in sessions.items():
            # Добавляем в батч для эмбеддингов
            from .filters.session_grouper import extract_session_content
            session_text = extract_session_content(session_messages)
            
            if session_text:
                # Батчевая обработка эмбеддингов
                should_process = self.batch_processor.add_embedding_task(
                    f"{dialogue_id}_{session_id}", session_text
                )
                
                if should_process:
                    self._process_embedding_batch()
                
                # Батчевое извлечение фактов
                if self.config.use_fact_extraction:
                    should_process = self.batch_processor.add_fact_extraction_task(
                        session_text, session_id, dialogue_id
                    )
                    
                    if should_process:
                        future = self.executor.submit(self._process_fact_batch)
                        futures.append(future)
                
                # Батчевое сжатие
                if self.config.use_compression:
                    should_process = self.batch_processor.add_compression_task(
                        f"{dialogue_id}_{session_id}", session_text
                    )
                    
                    if should_process:
                        future = self.executor.submit(self._process_compression_batch)
                        futures.append(future)
        
        # Ждем завершения всех задач
        for future in futures:
            future.result()
        
        # 3. Индексируем с векторным поиском
        index_stats = self.rag_engine.index_dialogue_compressed(dialogue_id, sessions)
        stats.update(index_stats)
        
        # 4. Сохраняем в кэш
        stats['processing_time'] = time.time() - start_time
        self.cache_manager.put(cache_key, stats, cache_type="general", ttl=3600)
        
        # Обновляем метрики
        self._update_metrics(stats)
        
        logger.info(f"Диалог {dialogue_id} обработан за {stats['processing_time']:.2f} сек")
        return stats
    
    def answer_question_optimized(self, dialogue_id: str, question: str) -> str:
        """
        Оптимизированный ответ на вопрос с кэшированием
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            
        Returns:
            Ответ на вопрос
        """
        start_time = time.time()
        
        # Проверяем кэш ответов
        cache_key = f"answer_{dialogue_id}_{hash(question)}"
        cached_answer = self.cache_manager.get(cache_key, cache_type="query")
        if cached_answer:
            logger.info(f"Ответ найден в кэше для вопроса: {question[:50]}...")
            self.metrics['total_queries'] += 1
            return cached_answer
        
        try:
            # Генерируем ответ
            answer = self._generate_answer(dialogue_id, question)
            
            # Сохраняем в кэш
            self.cache_manager.put(cache_key, answer, cache_type="query", ttl=1800)
            
            # Обновляем метрики
            query_time = time.time() - start_time
            self._update_query_metrics(query_time)
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при ответе на вопрос: {e}")
            return "Произошла ошибка при обработке вашего вопроса."
    
    def _generate_answer(self, dialogue_id: str, question: str) -> str:
        """Генерирует ответ на вопрос"""
        # Используем существующую логику из SmartMemory
        # но с оптимизациями
        
        # Получаем факты из кэша если возможно
        facts_key = f"facts_{dialogue_id}_{hash(question)}"
        facts = self.cache_manager.get(facts_key, cache_type="fact")
        
        if not facts and self.fact_database:
            facts = self.fact_database.query_facts(
                dialogue_id,
                query=question,
                min_confidence=self.config.fact_min_confidence
            )
            self.cache_manager.put(facts_key, facts, cache_type="fact")
        
        # Генерируем промпт через RAG
        prompt, metadata = self.rag_engine.process_question(question, dialogue_id, [])
        
        # Добавляем факты
        if facts:
            facts_text = "\n".join([f"• {fact.to_natural_text()}" for fact in facts[:5]])
            prompt = f"ФАКТЫ:\n{facts_text}\n\n{prompt}"
        
        # Генерируем ответ
        context_message = Message('system', prompt)
        answer = self.model.inference([context_message])
        
        return answer.strip()
    
    def _process_embedding_batch(self):
        """Обрабатывает батч эмбеддингов"""
        batch = self.batch_processor.get_embedding_batch()
        if not batch:
            return
        
        # Извлекаем тексты
        ids, texts = zip(*batch)
        
        # Создаем эмбеддинги батчем
        embeddings = self.rag_engine.embedding_engine.encode(list(texts))
        
        # Сохраняем в кэш
        for task_id, embedding in zip(ids, embeddings):
            self.cache_manager.put(task_id, embedding, cache_type="embedding")
    
    def _process_fact_batch(self):
        """Обрабатывает батч извлечения фактов"""
        batch = self.batch_processor.get_fact_batch()
        if not batch:
            return
        
        for text, session_id, dialogue_id in batch:
            # Проверяем кэш
            cache_key = f"facts_{session_id}_{hash(text)}"
            facts = self.cache_manager.get(cache_key, cache_type="fact")
            
            if not facts:
                facts = self.fact_extractor.extract_facts_from_text(
                    text, session_id, dialogue_id
                )
                self.cache_manager.put(cache_key, facts, cache_type="fact")
            
            # Добавляем в базу
            if facts:
                self.fact_database.add_facts(dialogue_id, facts)
    
    def _process_compression_batch(self):
        """Обрабатывает батч сжатия"""
        batch = self.batch_processor.get_compression_batch()
        if not batch:
            return
        
        for task_id, text in batch:
            # Проверяем кэш
            cache_key = f"comp_{task_id}"
            compressed = self.cache_manager.get(cache_key, cache_type="compression")
            
            if not compressed:
                result = self.rag_engine.compressor.compress(text)
                compressed = result.compressed_text
                self.cache_manager.put(cache_key, compressed, cache_type="compression")
    
    def _start_background_processing(self):
        """Запускает фоновую обработку батчей"""
        def background_worker():
            while True:
                try:
                    # Проверяем очереди
                    queue_sizes = self.batch_processor.get_queue_sizes()
                    
                    # Обрабатываем если есть данные
                    if queue_sizes['embedding_queue'] > 0:
                        self._process_embedding_batch()
                    
                    if queue_sizes['fact_queue'] > 0:
                        self._process_fact_batch()
                    
                    if queue_sizes['compression_queue'] > 0:
                        self._process_compression_batch()
                    
                    time.sleep(0.1)  # Небольшая задержка
                    
                except Exception as e:
                    logger.error(f"Ошибка в фоновой обработке: {e}")
                    time.sleep(1)
        
        self.background_thread = threading.Thread(target=background_worker, daemon=True)
        self.background_thread.start()
        logger.info("Фоновая обработка запущена")
    
    def _start_autosave(self):
        """Запускает автосохранение состояния"""
        interval = self.system_config.persistence_config.get('autosave_interval', 300)
        
        def autosave_worker():
            while True:
                try:
                    time.sleep(interval)
                    self.save_state()
                    logger.info("Автосохранение выполнено")
                except Exception as e:
                    logger.error(f"Ошибка автосохранения: {e}")
        
        self.autosave_thread = threading.Thread(target=autosave_worker, daemon=True)
        self.autosave_thread.start()
        logger.info(f"Автосохранение запущено (интервал: {interval} сек)")
    
    def _update_metrics(self, stats: Dict[str, Any]):
        """Обновляет метрики системы"""
        if 'compression' in stats and 'ratio' in stats['compression']:
            self.metrics['compression_ratio'] = stats['compression']['ratio']
        
        if 'facts_extracted' in stats:
            self.metrics['facts_per_dialogue'] = stats['facts_extracted']
    
    def _update_query_metrics(self, query_time: float):
        """Обновляет метрики запросов"""
        self.metrics['total_queries'] += 1
        
        # Скользящее среднее времени ответа
        alpha = 0.1
        self.metrics['avg_query_time'] = (
            alpha * query_time + 
            (1 - alpha) * self.metrics['avg_query_time']
        )
        
        # Обновляем cache hit rate
        cache_stats = self.cache_manager.get_stats()
        self.metrics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
    
    def save_state(self, save_dir: Optional[str] = None):
        """Сохраняет состояние системы"""
        save_dir = save_dir or self.system_config.persistence_config.get('save_dir', './gigamemory_state')
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Сохраняем компоненты
        self.rag_engine.save_indices(str(save_path / "vectors"))
        
        if self.fact_database:
            self.fact_database.save(str(save_path / "facts.json"))
        
        self.cache_manager.save(str(save_path / "cache.pkl"))
        
        # Сохраняем метрики
        import json
        with open(save_path / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Состояние сохранено в {save_dir}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Возвращает полную статистику системы"""
        return {
            'metrics': self.metrics,
            'cache': self.cache_manager.get_stats(),
            'queues': self.batch_processor.get_queue_sizes(),
            'rag': self.rag_engine.get_stats(),
            'compression': self.rag_engine.get_compression_stats()
        }
