"""
SmartMemory - интеллектуальная память с векторным поиском, извлечением фактов и сжатием
"""
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

from models import Message
from submit_interface import ModelWithMemory

from .llm_inference import ModelInference
from .modules.rag.compressed_rag_engine import CompressedRAGEngine, CompressedRAGConfig
from .modules.rag.fact_based_rag import FactBasedRAGEngine
from .modules.extraction.fact_database import FactDatabase
from .modules.extraction.fact_extractor import SmartFactExtractor
from .modules.compression.compression_models import CompressionLevel, CompressionMethod

logger = logging.getLogger(__name__)


class SmartMemoryConfig:
    """Конфигурация умной памяти"""
    
    # Векторный поиск
    use_vector_search: bool = True
    embedding_model: str = "cointegrated/rubert-tiny2"
    vector_top_k: int = 5
    
    # Извлечение фактов  
    use_fact_extraction: bool = True
    fact_min_confidence: float = 0.5
    
    # Сжатие
    use_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    
    # Интеграция
    use_hybrid_search: bool = True
    use_fact_based_rag: bool = True  # Использовать факт-ориентированный RAG
    max_context_length: int = 2000


class SmartMemory:
    """Интеллектуальная память с полной интеграцией компонентов"""
    
    def __init__(self, model_path: str, config: Optional[SmartMemoryConfig] = None):
        """
        Инициализация умной памяти
        
        Args:
            model_path: Путь к модели
            config: Конфигурация
        """
        self.config = config or SmartMemoryConfig()
        
        # Инициализируем модель
        self.model = ModelInference(model_path)
        
        # Создаем конфигурацию для RAG движка
        rag_config = CompressedRAGConfig(
            # Векторный поиск
            embedding_model=self.config.embedding_model,
            vector_search_top_k=self.config.vector_top_k,
            use_hybrid_search=self.config.use_hybrid_search,
            
            # Сжатие
            enable_compression=self.config.use_compression,
            compression_level=self.config.compression_level,
            
            # Общие настройки
            max_relevant_sessions=self.config.vector_top_k
        )
        
        # Создаем RAG движок со всеми компонентами
        self.rag_engine = CompressedRAGEngine(rag_config)
        
        # Извлекатель фактов
        if self.config.use_fact_extraction:
            self.fact_extractor = SmartFactExtractor(self.model, use_rules_first=True)
            self.fact_database = FactDatabase(conflict_strategy="latest")
        else:
            self.fact_extractor = None
            self.fact_database = None
        
        # Факт-ориентированный RAG движок
        if self.config.use_fact_based_rag and self.fact_database:
            self.fact_rag_engine = FactBasedRAGEngine(self.fact_database)
        else:
            self.fact_rag_engine = None
        
        # Статистика
        self.stats = {
            'dialogues_processed': 0,
            'facts_extracted': 0,
            'vectors_indexed': 0,
            'compression_ratio': 0.0,
            'queries_answered': 0
        }
        
        logger.info("SmartMemory инициализирована со всеми компонентами")
    
    def process_dialogue(self, dialogue_id: str, messages: List[Message]) -> Dict[str, Any]:
        """
        Обрабатывает диалог полным циклом
        
        Args:
            dialogue_id: ID диалога
            messages: Сообщения диалога
            
        Returns:
            Статистика обработки
        """
        start_time = datetime.now()
        stats = {
            'dialogue_id': dialogue_id,
            'messages_count': len(messages),
            'sessions_count': 0,
            'facts_extracted': 0,
            'vectors_indexed': 0,
            'compression_ratio': 1.0
        }
        
        # 1. Группируем по сессиям
        from .modules.storage.filters.session_grouper import SessionGrouper
        grouper = SessionGrouper()
        sessions = grouper.group_messages_by_sessions(messages, dialogue_id)
        stats['sessions_count'] = len(sessions)
        
        # 2. Извлекаем факты (если включено)
        if self.config.use_fact_extraction and self.fact_extractor:
            all_facts = []
            for session_id, session_messages in sessions.items():
                from .modules.storage.filters.session_grouper import extract_session_content
                session_text = extract_session_content(session_messages)
                
                if session_text:
                    try:
                        facts = self.fact_extractor.extract_facts_from_text(
                            session_text, session_id, dialogue_id
                        )
                        all_facts.extend(facts)
                    except Exception as e:
                        logger.warning(f"Ошибка извлечения фактов: {e}")
            
            # Сохраняем факты в базу
            if all_facts:
                self.fact_database.add_facts(dialogue_id, all_facts)
                stats['facts_extracted'] = len(all_facts)
                logger.info(f"Извлечено {len(all_facts)} фактов из диалога {dialogue_id}")
        
        # 3. Индексируем с векторным поиском и сжатием
        if self.config.use_vector_search:
            index_stats = self.rag_engine.index_dialogue_compressed(dialogue_id, sessions)
            stats['vectors_indexed'] = index_stats.get('sessions_indexed', 0)
            
            if 'compression' in index_stats:
                stats['compression_ratio'] = index_stats['compression'].get('ratio', 1.0)
        
        # 4. Обновляем глобальную статистику
        self.stats['dialogues_processed'] += 1
        self.stats['facts_extracted'] += stats['facts_extracted']
        self.stats['vectors_indexed'] += stats['vectors_indexed']
        
        # Вычисляем время обработки
        stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Диалог {dialogue_id} обработан за {stats['processing_time']:.2f} сек")
        return stats
    
    def answer_question(self, dialogue_id: str, question: str) -> str:
        """
        Отвечает на вопрос используя все компоненты
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            
        Returns:
            Ответ на вопрос
        """
        try:
            # Получаем данные диалога
            memory_data = self._get_dialogue_memory(dialogue_id)
            
            if not memory_data:
                return "У меня нет информации для ответа на этот вопрос."
            
            # Используем факт-ориентированный RAG если доступен
            if self.config.use_fact_based_rag and self.fact_rag_engine:
                prompt, metadata = self.fact_rag_engine.process_question(question, dialogue_id)
                
                # Логируем для отладки
                logger.info(f"Вопрос: {question}")
                logger.info(f"Определен тип: {metadata.get('fact_type')}")
                logger.info(f"Найдено фактов: {metadata.get('facts_found')}")
                
                # Если фактов нет, возвращаем готовый ответ
                if metadata['strategy'] == 'no_info':
                    return prompt
                else:
                    # Иначе используем модель
                    context_message = Message('system', prompt)
                    return self.model.inference([context_message])
            
            # Fallback к старому подходу
            # 1. Получаем релевантные факты
            facts_context = ""
            if self.config.use_fact_extraction and self.fact_database:
                facts = self.fact_database.query_facts(
                    dialogue_id, 
                    query=question,
                    min_confidence=self.config.fact_min_confidence
                )
                
                if facts:
                    facts_lines = []
                    for fact in facts[:5]:  # Топ-5 фактов
                        facts_lines.append(f"• {fact.to_natural_text()}")
                    facts_context = "ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:\n" + "\n".join(facts_lines) + "\n\n"
            
            # 2. Используем RAG движок для генерации промпта
            prompt, metadata = self.rag_engine.process_question(question, dialogue_id, memory_data)
            
            # 3. Добавляем факты в начало промпта
            if facts_context:
                prompt = facts_context + prompt
            
            # 4. Ограничиваем длину контекста
            if len(prompt) > self.config.max_context_length:
                prompt = self._truncate_context(prompt, self.config.max_context_length)
            
            # 5. Генерируем ответ через модель
            context_message = Message('system', prompt)
            answer = self.model.inference([context_message])
            
            # Обновляем статистику
            self.stats['queries_answered'] += 1
            
            logger.info(f"Ответ на вопрос '{question}' сгенерирован")
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при ответе на вопрос: {e}")
            return "Произошла ошибка при обработке вашего вопроса."
    
    def _get_dialogue_memory(self, dialogue_id: str) -> List[Message]:
        """
        Получает память диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Список сообщений
        """
        # Здесь должна быть логика получения сообщений из хранилища
        # Для примера возвращаем пустой список
        return []
    
    def _truncate_context(self, context: str, max_length: int) -> str:
        """
        Обрезает контекст до максимальной длины
        
        Args:
            context: Исходный контекст
            max_length: Максимальная длина
            
        Returns:
            Обрезанный контекст
        """
        if len(context) <= max_length:
            return context
        
        # Обрезаем и добавляем многоточие
        return context[:max_length-3] + "..."
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы"""
        stats = self.stats.copy()
        
        # Добавляем статистику компонентов
        if self.config.use_vector_search:
            stats['vector_search'] = self.rag_engine.get_stats()
        
        if self.config.use_fact_extraction and self.fact_database:
            stats['facts'] = self.fact_database.get_stats().to_dict()
        
        if self.config.use_compression:
            stats['compression'] = self.rag_engine.get_compression_stats()
        
        return stats
    
    def save_state(self, save_dir: str):
        """
        Сохраняет состояние системы
        
        Args:
            save_dir: Директория для сохранения
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Сохраняем векторные индексы
        if self.config.use_vector_search:
            self.rag_engine.save_indices(str(save_path / "vectors"))
        
        # Сохраняем базу фактов
        if self.config.use_fact_extraction and self.fact_database:
            self.fact_database.save(str(save_path / "facts.json"))
        
        logger.info(f"Состояние системы сохранено в {save_dir}")
    
    def load_state(self, save_dir: str):
        """
        Загружает состояние системы
        
        Args:
            save_dir: Директория с сохраненным состоянием
        """
        save_path = Path(save_dir)
        
        # Загружаем векторные индексы
        if self.config.use_vector_search:
            vectors_dir = save_path / "vectors"
            if vectors_dir.exists():
                self.rag_engine.load_indices(str(vectors_dir))
        
        # Загружаем базу фактов
        if self.config.use_fact_extraction and self.fact_database:
            facts_file = save_path / "facts.json"
            if facts_file.exists():
                self.fact_database.load(str(facts_file))
        
        logger.info(f"Состояние системы загружено из {save_dir}")
