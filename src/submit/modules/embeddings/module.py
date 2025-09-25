# modules/embeddings/module.py
from core.interfaces import IEmbeddingEngine, ProcessingResult
from typing import Dict, Any, List, Optional
import numpy as np
import hashlib

class EmbeddingsModule(IEmbeddingEngine):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'cointegrated/rubert-tiny2')
        
        # Импортируем только при необходимости
        from .embedding_engine import EmbeddingEngine
        from .vector_store import VectorStore
        
        self.engine = EmbeddingEngine(self.model_name, config.get('device', 'cuda'))
        self.vector_store = VectorStore(
            metric=config.get('metric', 'cosine'),
            index_type=config.get('index_type', 'flat')
        )
        
        # Оптимизатор будет установлен позже через set_optimizer
        self.optimizer = None
        
    def set_optimizer(self, optimizer):
        """ВАЖНО: Устанавливает оптимизатор для кэширования"""
        self.optimizer = optimizer
        if optimizer:
            # Включаем оптимизацию для эмбеддингов
            optimizer.optimize_for_embeddings()
    
    def encode_texts(self, texts: List[str]) -> ProcessingResult:
        """Кодирует тексты в векторы с кэшированием"""
        try:
            # Создаем ключ кэша из первых 3 текстов
            if self.optimizer and texts:
                cache_key = self._create_cache_key('encode', texts[:3])
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True, 'count': len(cached)}
                    )
            
            # Батчевая обработка через оптимизатор если доступен
            if self.optimizer and len(texts) > 10:
                tasks = [{'text': text, 'idx': i} for i, text in enumerate(texts)]
                batch_result = self.optimizer.batch_process(
                    tasks,
                    lambda task: self.engine.encode_single(task['text'])
                )
                if batch_result.success:
                    vectors = batch_result.data
                else:
                    vectors = self.engine.encode_batch(texts)
            else:
                vectors = self.engine.encode_batch(texts)
            
            # Кэшируем результат
            if self.optimizer and vectors:
                self.optimizer.cache_put(cache_key, vectors, ttl=7200)
            
            return ProcessingResult(
                success=True,
                data=vectors,
                metadata={'encoded_count': len(vectors), 'model': self.model_name}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                error=f"Encoding failed: {str(e)}"
            )
    
    def vector_search(self, query: str, dialogue_id: str, top_k: int = 5) -> ProcessingResult:
        """Поиск по векторам с кэшированием результатов"""
        try:
            # Проверяем кэш
            if self.optimizer:
                cache_key = self._create_cache_key('search', dialogue_id, query, top_k)
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True, 'dialogue_id': dialogue_id}
                    )
            
            # Кодируем запрос
            query_vector = self.engine.encode_single(query)
            
            # Ищем похожие
            results = self.vector_store.search(
                dialogue_id=dialogue_id,
                query_vector=query_vector,
                top_k=top_k
            )
            
            # Кэшируем на 30 минут
            if self.optimizer and results:
                self.optimizer.cache_put(cache_key, results, ttl=1800)
            
            return ProcessingResult(
                success=True,
                data=results,
                metadata={
                    'query': query,
                    'dialogue_id': dialogue_id,
                    'found_count': len(results)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                error=f"Search failed: {str(e)}"
            )
    
    def index_dialogue(self, dialogue_id: str, sessions: Dict[str, List]) -> ProcessingResult:
        """Индексирует диалог для поиска"""
        try:
            indexed_count = 0
            
            for session_id, messages in sessions.items():
                # Извлекаем тексты
                texts = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        texts.append(msg.content)
                    elif isinstance(msg, dict) and 'content' in msg:
                        texts.append(msg['content'])
                
                if not texts:
                    continue
                
                # Кодируем тексты
                encode_result = self.encode_texts(texts)
                if not encode_result.success:
                    continue
                
                vectors = encode_result.data
                
                # Добавляем в хранилище
                self.vector_store.add_vectors(
                    dialogue_id=dialogue_id,
                    session_id=session_id,
                    vectors=vectors,
                    texts=texts
                )
                indexed_count += len(vectors)
            
            return ProcessingResult(
                success=True,
                data={'indexed_count': indexed_count},
                metadata={
                    'dialogue_id': dialogue_id,
                    'sessions_count': len(sessions)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={'indexed_count': 0},
                error=f"Indexing failed: {str(e)}"
            )
    
    # Дополнительные методы для интеграции
    def hybrid_search(self, query: str, dialogue_id: str, keywords: List[str]) -> ProcessingResult:
        """Комбинированный поиск: векторный + keyword"""
        # Векторный поиск
        vector_results = self.vector_search(query, dialogue_id, top_k=10)
        
        if not vector_results.success:
            return vector_results
        
        # Фильтруем по ключевым словам
        filtered = []
        for result in vector_results.data:
            text = result.get('text', '').lower()
            if any(kw.lower() in text for kw in keywords):
                result['boost'] = 1.2  # Увеличиваем релевантность
                filtered.append(result)
        
        # Добавляем остальные результаты
        for result in vector_results.data:
            if result not in filtered:
                filtered.append(result)
        
        return ProcessingResult(
            success=True,
            data=filtered[:5],
            metadata={'hybrid': True, 'keywords': keywords}
        )
    
    def save_index(self, dialogue_id: str, filepath: str):
        """Сохраняет индекс на диск"""
        return self.vector_store.save(dialogue_id, filepath)
    
    def load_index(self, dialogue_id: str, filepath: str):
        """Загружает индекс с диска"""
        return self.vector_store.load(dialogue_id, filepath)
    
    def _create_cache_key(self, *args) -> str:
        """Создает ключ для кэша"""
        key_str = '_'.join(str(arg)[:50] for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()