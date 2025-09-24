"""
Факт-ориентированный RAG движок для работы с извлеченными фактами
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .interface import RAGInterface
from ..extraction.fact_database import FactDatabase
from ..extraction.fact_models import Fact
from ..embeddings.improved_vector_store import ImprovedVectorStore
from ..llm_inference import LLMInference


@dataclass
class FactBasedRAGConfig:
    """Конфигурация для факт-ориентированного RAG"""
    max_facts: int = 10
    similarity_threshold: float = 0.7
    use_embeddings: bool = True
    use_llm_reranking: bool = True
    fact_weight_decay: float = 0.1  # Затухание веса фактов со временем
    min_fact_confidence: float = 0.5  # Минимальная уверенность в факте


class FactBasedRAGEngine(RAGInterface):
    """Факт-ориентированный RAG движок"""
    
    def __init__(self, 
                 fact_database: FactDatabase,
                 vector_store: Optional[ImprovedVectorStore] = None,
                 llm_inference: Optional[LLMInference] = None,
                 config: Optional[FactBasedRAGConfig] = None):
        """
        Инициализация факт-ориентированного RAG движка
        
        Args:
            fact_database: База данных фактов
            vector_store: Векторное хранилище для поиска
            llm_inference: LLM для переранжирования
            config: Конфигурация движка
        """
        self.fact_database = fact_database
        self.vector_store = vector_store
        self.llm_inference = llm_inference
        self.config = config or FactBasedRAGConfig()
        
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """
        Поиск релевантных фактов для запроса
        
        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            
        Returns:
            Список релевантных фактов с метаданными
        """
        max_results = max_results or self.config.max_facts
        
        # Получаем все факты из базы данных
        all_facts = self.fact_database.get_all_facts()
        
        if not all_facts:
            self.logger.warning("База данных фактов пуста")
            return []
        
        # Фильтруем факты по минимальной уверенности
        filtered_facts = [
            fact for fact in all_facts 
            if fact.confidence >= self.config.min_fact_confidence
        ]
        
        if not filtered_facts:
            self.logger.warning("Нет фактов с достаточной уверенностью")
            return []
        
        # Если есть векторное хранилище, используем его для поиска
        if self.config.use_embeddings and self.vector_store:
            relevant_facts = self._search_with_embeddings(query, filtered_facts, max_results)
        else:
            # Простой поиск по ключевым словам
            relevant_facts = self._search_with_keywords(query, filtered_facts, max_results)
        
        # Применяем затухание веса по времени
        relevant_facts = self._apply_time_decay(relevant_facts)
        
        # Если включено переранжирование через LLM
        if self.config.use_llm_reranking and self.llm_inference:
            relevant_facts = self._rerank_with_llm(query, relevant_facts)
        
        # Конвертируем в формат результата
        results = []
        for fact in relevant_facts[:max_results]:
            result = {
                "content": fact.content,
                "confidence": fact.confidence,
                "source": fact.source,
                "timestamp": fact.timestamp.isoformat() if fact.timestamp else None,
                "metadata": fact.metadata or {},
                "fact_id": fact.fact_id,
                "relevance_score": getattr(fact, 'relevance_score', 0.0)
            }
            results.append(result)
        
        return results
    
    def _search_with_embeddings(self, query: str, facts: List[Fact], max_results: int) -> List[Fact]:
        """Поиск с использованием векторных эмбеддингов"""
        try:
            # Получаем эмбеддинг запроса
            query_embedding = self.vector_store.encode_text(query)
            
            # Вычисляем схожесть с каждым фактом
            for fact in facts:
                if hasattr(fact, 'embedding') and fact.embedding:
                    similarity = self.vector_store.cosine_similarity(query_embedding, fact.embedding)
                    fact.relevance_score = similarity
                else:
                    # Если у факта нет эмбеддинга, используем простое совпадение по словам
                    fact.relevance_score = self._calculate_keyword_similarity(query, fact.content)
            
            # Фильтруем по порогу схожести и сортируем
            relevant_facts = [
                fact for fact in facts 
                if fact.relevance_score >= self.config.similarity_threshold
            ]
            relevant_facts.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return relevant_facts
            
        except Exception as e:
            self.logger.error(f"Ошибка при поиске с эмбеддингами: {e}")
            # Fallback к поиску по ключевым словам
            return self._search_with_keywords(query, facts, max_results)
    
    def _search_with_keywords(self, query: str, facts: List[Fact], max_results: int) -> List[Fact]:
        """Простой поиск по ключевым словам"""
        query_words = set(query.lower().split())
        
        for fact in facts:
            fact_words = set(fact.content.lower().split())
            # Вычисляем Jaccard similarity
            intersection = len(query_words.intersection(fact_words))
            union = len(query_words.union(fact_words))
            similarity = intersection / union if union > 0 else 0.0
            fact.relevance_score = similarity
        
        # Сортируем по релевантности
        relevant_facts = sorted(facts, key=lambda x: x.relevance_score, reverse=True)
        
        return relevant_facts
    
    def _calculate_keyword_similarity(self, query: str, content: str) -> float:
        """Вычисляет схожесть по ключевым словам"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_time_decay(self, facts: List[Fact]) -> List[Fact]:
        """Применяет затухание веса фактов по времени"""
        if self.config.fact_weight_decay <= 0:
            return facts
        
        current_time = datetime.now()
        
        for fact in facts:
            if fact.timestamp:
                # Вычисляем возраст факта в днях
                age_days = (current_time - fact.timestamp).days
                # Применяем затухание
                decay_factor = 1.0 / (1.0 + self.config.fact_weight_decay * age_days)
                fact.relevance_score *= decay_factor
        
        # Пересортируем с учетом затухания
        facts.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return facts
    
    def _rerank_with_llm(self, query: str, facts: List[Fact]) -> List[Fact]:
        """Переранжирует факты с помощью LLM"""
        if not self.llm_inference or len(facts) <= 1:
            return facts
        
        try:
            # Создаем промпт для переранжирования
            facts_text = "\n\n".join([
                f"Факт {i+1}: {fact.content}\nУверенность: {fact.confidence:.2f}\nРелевантность: {fact.relevance_score:.2f}"
                for i, fact in enumerate(facts)
            ])
            
            prompt = f"""
Запрос пользователя: {query}

Факты для ранжирования:
{facts_text}

Пожалуйста, переранжируйте факты по релевантности к запросу пользователя.
Верните только номера фактов в порядке убывания релевантности, разделенные запятыми.
Например: 3,1,5,2,4
"""
            
            response = self.llm_inference.generate_response(prompt)
            
            # Парсим ответ и переранжируем
            try:
                rankings = [int(x.strip()) - 1 for x in response.split(',')]
                reranked_facts = []
                
                for rank in rankings:
                    if 0 <= rank < len(facts):
                        reranked_facts.append(facts[rank])
                
                # Добавляем оставшиеся факты, если они не были упомянуты
                for i, fact in enumerate(facts):
                    if i not in rankings:
                        reranked_facts.append(fact)
                
                return reranked_facts
                
            except (ValueError, IndexError):
                self.logger.warning("Не удалось распарсить ответ LLM для переранжирования")
                return facts
                
        except Exception as e:
            self.logger.error(f"Ошибка при переранжировании через LLM: {e}")
            return facts
    
    def get_fact_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Получает факт по ID"""
        fact = self.fact_database.get_fact_by_id(fact_id)
        if not fact:
            return None
        
        return {
            "content": fact.content,
            "confidence": fact.confidence,
            "source": fact.source,
            "timestamp": fact.timestamp.isoformat() if fact.timestamp else None,
            "metadata": fact.metadata or {},
            "fact_id": fact.fact_id
        }
    
    def add_fact(self, content: str, source: str, confidence: float = 1.0, 
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Добавляет новый факт в базу данных"""
        fact_id = self.fact_database.add_fact(
            content=content,
            source=source,
            confidence=confidence,
            metadata=metadata
        )
        
        # Если есть векторное хранилище, добавляем эмбеддинг
        if self.vector_store:
            try:
                embedding = self.vector_store.encode_text(content)
                self.fact_database.update_fact_embedding(fact_id, embedding)
            except Exception as e:
                self.logger.error(f"Ошибка при создании эмбеддинга для факта {fact_id}: {e}")
        
        return fact_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Получает статистику по базе фактов"""
        all_facts = self.fact_database.get_all_facts()
        
        if not all_facts:
            return {
                "total_facts": 0,
                "avg_confidence": 0.0,
                "sources": {},
                "topics": {}
            }
        
        # Общая статистика
        total_facts = len(all_facts)
        avg_confidence = sum(fact.confidence for fact in all_facts) / total_facts
        
        # Статистика по источникам
        sources = {}
        for fact in all_facts:
            source = fact.source
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        # Статистика по темам (если есть в метаданных)
        topics = {}
        for fact in all_facts:
            if fact.metadata and 'topic' in fact.metadata:
                topic = fact.metadata['topic']
                if topic not in topics:
                    topics[topic] = 0
                topics[topic] += 1
        
        return {
            "total_facts": total_facts,
            "avg_confidence": avg_confidence,
            "sources": sources,
            "topics": topics
        }