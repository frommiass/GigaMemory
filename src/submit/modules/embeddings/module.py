# modules/embeddings/module.py
"""
Интеграционный модуль Embeddings - ТОЛЬКО КООРДИНАЦИЯ!
"""

from core.interfaces import IEmbeddingEngine, ProcessingResult
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import hashlib
import re
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EmbeddingsModule(IEmbeddingEngine):
    """Модуль эмбеддингов - координирует engine и store"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'cointegrated/rubert-tiny2')
        
        # Компоненты инициализируем лениво
        self.engine = None
        self.vector_store = None
        self.optimizer = None
        
        # Параметры фильтрации
        self.max_text_length = 2000
        self.min_entropy = 0.3
        
        # Расширяем паттерны личной информации
        self.personal_patterns = [
            # Имена и представления
            r'меня зовут\s+(\w+)', r'моё? имя\s+(\w+)', r'я\s+(\w+)',
            r'my name is\s+(\w+)', r'i am\s+(\w+)', r'call me\s+(\w+)',
            
            # Возраст
            r'мне\s+(\d+)', r'(\d+)\s+лет', r'(\d+)\s+года?', r'возраст\s+(\d+)',
            r'i am\s+(\d+)\s+years', r'age\s+(\d+)',
            
            # Работа и профессия
            r'работаю\s+([\w\s]+)', r'я\s+([\w\s]+)\s+по профессии',
            r'профессия\s+([\w\s]+)', r'должность\s+([\w\s]+)',
            r'i work as\s+([\w\s]+)', r'occupation\s+([\w\s]+)',
            
            # Местоположение
            r'живу в\s+([\w\s]+)', r'из\s+([\w\s]+)', r'родом из\s+([\w\s]+)',
            r'переехал в\s+([\w\s]+)', r'нахожусь в\s+([\w\s]+)',
            
            # Семья
            r'женат', r'замужем', r'холост', r'разведен',
            r'(\d+)\s+дет', r'дочь', r'сын', r'жена', r'муж',
            
            # Образование
            r'учусь в\s+([\w\s]+)', r'окончил\s+([\w\s]+)',
            r'студент\s+([\w\s]+)', r'выпускник\s+([\w\s]+)',
            
            # Интересы
            r'увлекаюсь\s+([\w\s]+)', r'люблю\s+([\w\s]+)',
            r'хобби\s+([\w\s]+)', r'интересуюсь\s+([\w\s]+)'
        ]
        
        # Стоп-фразы для детекции мусора
        self.stop_phrases = [
            'lorem ipsum', 'test test', '<!doctype', '<html',
            'import react', 'def __init__', 'class ', 'function(',
            'consectetur adipiscing', 'dolor sit amet'
        ]
        
        self.stats = {
            'indexed': 0,
            'filtered': 0,
            'searches': 0
        }
    
    def _lazy_init(self):
        """Ленивая инициализация компонентов"""
        if self.engine is None:
            from .embedding_engine import EmbeddingEngine
            from .improved_vector_store import ImprovedVectorStore
            
            self.engine = EmbeddingEngine(
                self.model_name,
                device=self.config.get('device', 'cuda')
            )
            
            # Используем улучшенное хранилище с FAISS
            self.vector_store = ImprovedVectorStore(
                use_faiss=self.config.get('use_faiss', True),
                metric=self.config.get('metric', 'cosine')
            )
    
    def set_dependencies(self, optimizer=None, storage=None, embeddings=None):
        """Установка оптимизатора"""
        self.optimizer = optimizer
    
    def should_index_text(self, text: str) -> bool:
        """
        Фильтрация мусора - КРИТИЧЕСКАЯ функция!
        Возвращает True если текст стоит индексировать
        """
        # Проверка длины
        if len(text) > self.max_text_length or len(text) < 10:
            self.stats['filtered'] += 1
            return False
        
        # Проверка на копипаст
        text_lower = text.lower()
        for stop in self.stop_phrases:
            if stop in text_lower:
                self.stats['filtered'] += 1
                return False
        
        # Проверка энтропии (повторяемость)
        entropy = self._calculate_entropy(text)
        if entropy < self.min_entropy:
            self.stats['filtered'] += 1
            return False
        
        # Проверка на личную информацию - всегда индексируем!
        for pattern in self.personal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Проверка на полезный контекст
        useful_words = [
            'я', 'мой', 'моя', 'меня', 'мне', 'мною',
            'хочу', 'буду', 'делаю', 'думаю', 'считаю',
            'нужно', 'важно', 'интересно', 'сложно'
        ]
        
        # Если нет полезных слов в длинном тексте - вероятно мусор
        if len(text) > 500:
            if not any(word in text_lower for word in useful_words):
                self.stats['filtered'] += 1
                return False
        
        return True
    
    def _calculate_entropy(self, text: str) -> float:
        """Вычисляет энтропию для определения повторяемости"""
        if not text:
            return 0.0
        
        # Считаем уникальные слова
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio
    
    def _smart_chunk(self, text: str, size: int = 300) -> List[str]:
        """Умное разбиение с оверлапом"""
        if len(text) <= size:
            return [text]
        
        # Разбиваем по предложениям
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            if current_len + len(sent) > size and current:
                chunks.append(' '.join(current))
                # Оверлап - берём последнее предложение
                if len(current) > 1:
                    current = [current[-1]]
                    current_len = len(current[0])
                else:
                    current = []
                    current_len = 0
            
            current.append(sent)
            current_len += len(sent)
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def _prioritize_sessions(self, sessions: Dict) -> Dict[str, float]:
        """Приоритизация сессий по важности"""
        priorities = {}
        
        for sid, messages in sessions.items():
            score = 0.0
            text_len = 0
            
            # Анализируем контент
            for msg in messages:
                content = self._extract_content(msg)
                text_len += len(content)
                
                # Проверяем личную информацию
                for pattern in self.personal_patterns[:10]:  # Топ паттерны
                    if re.search(pattern, content.lower()):
                        score += 1.0
                        break
            
            # Штраф за слишком длинные (копипаст)
            if text_len > 5000:
                score *= 0.5
            elif text_len < 100:
                score *= 0.8
            
            priorities[sid] = min(max(score / max(1, len(messages)), 0.0), 1.0)
        
        return priorities
    
    def _extract_content(self, msg) -> str:
        """Извлечение текста из сообщения"""
        if hasattr(msg, 'content'):
            return msg.content
        elif isinstance(msg, dict):
            return msg.get('content', '')
        return str(msg)
    
    def index_dialogue(self, dialogue_id: str, sessions: Dict[str, List]) -> ProcessingResult:
        """
        Индексация диалога с фильтрацией и оптимизацией
        """
        self._lazy_init()
        start_time = time.time()
        
        try:
            indexed = 0
            skipped = 0
            
            # Приоритизируем сессии
            priorities = self._prioritize_sessions(sessions)
            
            # Обрабатываем по приоритету
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: priorities.get(x[0], 0),
                reverse=True
            )
            
            all_texts = []
            all_metadata = []
            
            for session_id, messages in sorted_sessions:
                priority = priorities.get(session_id, 0)
                
                # Пропускаем неважные при большом объёме
                if len(sessions) > 50 and priority < 0.3:
                    skipped += len(messages)
                    continue
                
                for msg_idx, msg in enumerate(messages):
                    content = self._extract_content(msg)
                    
                    # ФИЛЬТРАЦИЯ!
                    if not self.should_index_text(content):
                        skipped += 1
                        continue
                    
                    # Чанки для длинных
                    if len(content) > 500:
                        chunks = self._smart_chunk(content)
                    else:
                        chunks = [content]
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        all_texts.append(chunk)
                        all_metadata.append({
                            'session_id': session_id,
                            'msg_idx': msg_idx,
                            'chunk_idx': chunk_idx,
                            'priority': priority
                        })
            
            # Батчевое кодирование
            if all_texts:
                if self.optimizer:
                    # Через оптимизатор с кэшем
                    cache_key = f"encode_{dialogue_id}"
                    vectors = self.optimizer.cache_get(cache_key)
                    
                    if vectors is None:
                        vectors = self.engine.encode_batch(all_texts)
                        self.optimizer.cache_put(cache_key, vectors, ttl=3600)
                else:
                    vectors = self.engine.encode_batch(all_texts)
                
                # Добавляем в хранилище
                self.vector_store.add_batch(
                    dialogue_id=dialogue_id,
                    vectors=vectors,
                    texts=all_texts,
                    metadata=all_metadata
                )
                
                indexed = len(vectors)
            
            elapsed = time.time() - start_time
            self.stats['indexed'] += indexed
            
            return ProcessingResult(
                success=True,
                data={'indexed': indexed, 'skipped': skipped, 'time': elapsed},
                metadata={'dialogue_id': dialogue_id}
            )
            
        except Exception as e:
            logger.error(f"Ошибка индексации: {e}")
            return ProcessingResult(success=False, data={}, error=str(e))
    
    def hybrid_search(self, query: str, dialogue_id: str, 
                     keywords: List[str] = None) -> ProcessingResult:
        """Гибридный поиск: векторы + ключевые слова"""
        self._lazy_init()
        
        try:
            # Извлекаем ключевые слова
            if not keywords:
                keywords = self._extract_keywords(query)
            
            # Векторный поиск
            query_vector = self.engine.encode_single(query)
            results = self.vector_store.search(
                dialogue_id=dialogue_id,
                query_vector=query_vector,
                top_k=15
            )
            
            # Ранжирование с учётом ключевых слов
            for result in results:
                score = result['score']
                text_lower = result['text'].lower()
                
                # Буст за ключевые слова
                for kw in keywords:
                    if kw.lower() in text_lower:
                        score *= 1.2
                
                # Буст за приоритет
                if result.get('metadata', {}).get('priority', 0) > 0.5:
                    score *= 1.1
                
                result['final_score'] = score
            
            # Сортировка и топ-5
            results.sort(key=lambda x: x['final_score'], reverse=True)
            final_results = results[:5]
            
            self.stats['searches'] += 1
            
            return ProcessingResult(
                success=True,
                data=final_results,
                metadata={'query': query, 'keywords': keywords}
            )
            
        except Exception as e:
            return ProcessingResult(success=False, data=[], error=str(e))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Извлечение ключевых слов"""
        stop_words = {'как', 'что', 'где', 'когда', 'почему', 'у', 'в', 'на', 'с'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    # Совместимость с интерфейсом
    def encode_texts(self, texts: List[str]) -> ProcessingResult:
        self._lazy_init()
        try:
            vectors = self.engine.encode_batch(texts)
            return ProcessingResult(success=True, data=vectors)
        except Exception as e:
            return ProcessingResult(success=False, data=None, error=str(e))
    
    def vector_search(self, query: str, dialogue_id: str, top_k: int = 5) -> ProcessingResult:
        return self.hybrid_search(query, dialogue_id)