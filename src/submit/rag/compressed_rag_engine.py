"""
RAG движок с интегрированным семантическим сжатием
"""
from typing import Dict, List, Tuple, Optional, Any
import logging
from models import Message

from .vector_rag_engine import VectorRAGEngine, VectorRAGConfig
from ..compression import (
    HierarchicalCompressor, 
    CompressionConfig, 
    CompressionLevel,
    CompressionMethod
)

logger = logging.getLogger(__name__)


class CompressedRAGConfig(VectorRAGConfig):
    """Конфигурация для RAG с сжатием"""
    
    # Настройки сжатия
    enable_compression: bool = True
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    compression_method: CompressionMethod = CompressionMethod.HYBRID
    compression_target_ratio: float = 0.3
    compression_min_length: int = 200
    use_hierarchical: bool = True
    
    # Кэширование сжатых данных
    cache_compressed_sessions: bool = True
    compressed_cache_size: int = 1000


class CompressedRAGEngine(VectorRAGEngine):
    """RAG движок с семантическим сжатием"""
    
    def __init__(self, config: Optional[CompressedRAGConfig] = None):
        """
        Инициализация RAG с сжатием
        
        Args:
            config: Конфигурация движка
        """
        self.config = config or CompressedRAGConfig()
        super().__init__(self.config)
        
        # Создаем компрессор
        compression_config = CompressionConfig(
            level=self.config.compression_level,
            method=self.config.compression_method,
            target_ratio=self.config.compression_target_ratio,
            min_length=self.config.compression_min_length,
            use_cache=self.config.cache_compressed_sessions
        )
        
        if self.config.use_hierarchical:
            self.compressor = HierarchicalCompressor(compression_config)
        else:
            from ..compression import SemanticCompressor
            self.compressor = SemanticCompressor(compression_config)
        
        # Кэш сжатых сессий
        self.compressed_cache: Dict[str, str] = {}
        
        logger.info(f"Инициализирован CompressedRAGEngine с уровнем сжатия {self.config.compression_level.value}")
    
    def _compress_session(self, session_id: str, messages: List[Message]) -> str:
        """
        Сжимает сессию
        
        Args:
            session_id: ID сессии
            messages: Сообщения сессии
            
        Returns:
            Сжатый текст сессии
        """
        # Проверяем кэш
        cache_key = f"{session_id}_{len(messages)}"
        if cache_key in self.compressed_cache:
            return self.compressed_cache[cache_key]
        
        # Извлекаем текст сессии
        from ..filters.session_grouper import extract_session_content
        session_text = extract_session_content(messages)
        
        if not session_text or len(session_text) < self.config.compression_min_length:
            # Не сжимаем короткие тексты
            compressed = session_text
        else:
            # Сжимаем текст
            if self.config.use_hierarchical:
                result = self.compressor.compress_hierarchically(session_text)
            else:
                result = self.compressor.compress(session_text)
            
            compressed = result.compressed_text
            
            logger.debug(f"Сессия {session_id} сжата: {len(session_text)} -> {len(compressed)} символов "
                        f"(коэффициент {result.compression_ratio:.2f})")
        
        # Сохраняем в кэш
        if self.config.cache_compressed_sessions and len(self.compressed_cache) < self.config.compressed_cache_size:
            self.compressed_cache[cache_key] = compressed
        
        return compressed
    
    def _compress_dialogue_sessions(self, dialogue_id: str, sessions: Dict[str, List[Message]]) -> Dict[str, Any]:
        """
        Сжимает все сессии диалога
        
        Args:
            dialogue_id: ID диалога
            sessions: Словарь сессий
            
        Returns:
            Статистика сжатия
        """
        if not self.config.enable_compression:
            # Если сжатие отключено, просто копируем исходные тексты
            self.compressed_sessions[dialogue_id] = {}
            for session_id, messages in sessions.items():
                session_content = extract_session_content(messages)
                self.compressed_sessions[dialogue_id][session_id] = session_content
            return {'compression_disabled': True, 'sessions_processed': len(sessions)}
        
        logger.info(f"Начинаем сжатие {len(sessions)} сессий для диалога {dialogue_id}")
        
        # Подготавливаем тексты для сжатия
        texts_to_compress = []
        session_ids = []
        
        for session_id, messages in sessions.items():
            session_content = extract_session_content(messages)
            if len(session_content) >= self.config.compression_min_length:
                texts_to_compress.append(session_content)
                session_ids.append(session_id)
        
        if not texts_to_compress:
            logger.warning("Нет сессий для сжатия (все слишком короткие)")
            return {'sessions_processed': 0, 'sessions_skipped': len(sessions)}
        
        # Выбираем стратегию сжатия
        if (self.config.enable_hierarchical_compression and 
            len(texts_to_compress) >= self.config.hierarchical_threshold):
            # Иерархическое сжатие для большого количества сессий
            compressed_results = self._compress_hierarchically(texts_to_compress, session_ids)
        else:
            # Обычное батчевое сжатие
            compressed_results = self._compress_batch(texts_to_compress, session_ids)
        
        # Сохраняем результаты
        self.compressed_sessions[dialogue_id] = {}
        stats = {
            'sessions_processed': 0,
            'sessions_skipped': len(sessions) - len(texts_to_compress),
            'total_original_chars': 0,
            'total_compressed_chars': 0,
            'average_compression_ratio': 0.0,
            'compression_method': 'hierarchical' if len(texts_to_compress) >= self.config.hierarchical_threshold else 'batch'
        }
        
        for session_id, result in zip(session_ids, compressed_results):
            if isinstance(result, CompressionResult) and result.is_valid():
                self.compressed_sessions[dialogue_id][session_id] = result.compressed_text
                stats['sessions_processed'] += 1
                stats['total_original_chars'] += result.original_length
                stats['total_compressed_chars'] += result.compressed_length
            else:
                # Fallback на исходный текст если сжатие не удалось
                original_text = next(text for text, sid in zip(texts_to_compress, session_ids) if sid == session_id)
                self.compressed_sessions[dialogue_id][session_id] = original_text
        
        # Добавляем несжатые короткие сессии
        for session_id, messages in sessions.items():
            if session_id not in self.compressed_sessions[dialogue_id]:
                session_content = extract_session_content(messages)
                self.compressed_sessions[dialogue_id][session_id] = session_content
        
        # Рассчитываем среднее сжатие
        if stats['total_original_chars'] > 0:
            stats['average_compression_ratio'] = stats['total_compressed_chars'] / stats['total_original_chars']
        
        return stats
    
    def _compress_batch(self, texts: List[str], session_ids: List[str]) -> List[CompressionResult]:
        """Батчевое сжатие текстов"""
        results = []
        
        for text, session_id in zip(texts, session_ids):
            try:
                result = self.compressor.compress(text)
                results.append(result)
                logger.debug(f"Сессия {session_id}: {result.original_length} -> {result.compressed_length} символов")
            except Exception as e:
                logger.error(f"Ошибка сжатия сессии {session_id}: {e}")
                # Создаем fallback результат
                results.append(CompressionResult(
                    original_text=text,
                    compressed_text=text,
                    compression_ratio=1.0,
                    method_used=self.config.compression_method,
                    level_used=self.config.compression_level
                ))
        
        return results
    
    def _compress_hierarchically(self, texts: List[str], session_ids: List[str]) -> List[CompressionResult]:
        """Иерархическое сжатие для большого количества сессий"""
        # Объединяем все тексты для иерархической обработки
        combined_text = "\n\n".join(f"[Сессия {sid}] {text}" for text, sid in zip(texts, session_ids))
        
        try:
            # Применяем иерархическое сжатие
            hierarchical_result = self.hierarchical_compressor.compress_hierarchically(combined_text)
            
            if hierarchical_result.is_valid():
                # Разбиваем результат обратно на сессии
                compressed_parts = hierarchical_result.compressed_text.split("[Сессия")
                
                results = []
                for i, (original_text, session_id) in enumerate(zip(texts, session_ids)):
                    if i + 1 < len(compressed_parts):
                        # Извлекаем сжатую часть для этой сессии
                        session_part = compressed_parts[i + 1].split("]", 1)
                        if len(session_part) > 1:
                            compressed_content = session_part[1].strip()
                        else:
                            compressed_content = original_text
                    else:
                        compressed_content = original_text
                    
                    # Создаем результат для сессии
                    session_result = CompressionResult(
                        original_text=original_text,
                        compressed_text=compressed_content,
                        compression_ratio=len(compressed_content) / len(original_text) if original_text else 1.0,
                        method_used=CompressionMethod.HYBRID,
                        level_used=self.config.compression_level
                    )
                    results.append(session_result)
                
                return results
        except Exception as e:
            logger.error(f"Ошибка иерархического сжатия: {e}")
        
        # Fallback на обычное батчевое сжатие
        return self._compress_batch(texts, session_ids)
    
    def _process_with_compressed_topic_rag(self, question: str, topic: str, confidence: float,
                                         dialogue_id: str) -> Tuple[str, Dict]:
        """Обрабатывает вопрос с тематическим RAG используя сжатые сессии"""
        
        # Получаем сжатые сессии
        compressed_sessions = self.compressed_sessions.get(dialogue_id, {})
        
        if not compressed_sessions:
            return "У меня нет информации для ответа на этот вопрос.", {}
        
        # Находим релевантные сессии через keyword matching на сжатых текстах
        relevant_session_ids = self._find_relevant_compressed_sessions(question, topic, compressed_sessions)
        
        # Ранжируем по релевантности
        ranked_sessions = self._rank_compressed_sessions(question, relevant_session_ids, compressed_sessions)
        
        # Выбираем топ сессии
        top_session_ids = ranked_sessions[:self.config.max_relevant_sessions]
        
        # Формируем контекст из сжатых сессий
        memory_text = self._build_compressed_memory_context(top_session_ids, compressed_sessions)
        
        # Генерируем тематический промпт
        prompt = get_topic_prompt(topic, question, memory_text)
        
        # Метаданные для отладки
        metadata = {
            'strategy': 'compressed_topic_rag',
            'topic': topic,
            'confidence': confidence,
            'total_sessions': len(compressed_sessions),
            'relevant_sessions': len(relevant_session_ids),
            'selected_sessions': len(top_session_ids),
            'selected_session_ids': top_session_ids,
            'memory_length': len(memory_text),
            'compression_enabled': self.config.enable_compression
        }
        
        return prompt, metadata
    
    def _process_with_compressed_fallback(self, question: str, dialogue_id: str) -> Tuple[str, Dict]:
        """Обрабатывает вопрос с fallback стратегией используя сжатые сессии"""
        
        # Получаем сжатые сессии
        compressed_sessions = self.compressed_sessions.get(dialogue_id, {})
        
        if not compressed_sessions:
            return "У меня нет информации для ответа на этот вопрос.", {}
        
        # Ранжируем все сессии по релевантности к вопросу
        all_session_ids = list(compressed_sessions.keys())
        ranked_sessions = self._rank_compressed_sessions(question, all_session_ids, compressed_sessions)
        
        # Выбираем топ сессии
        top_session_ids = ranked_sessions[:self.config.max_relevant_sessions]
        
        # Формируем контекст из сжатых сессий
        memory_text = self._build_compressed_memory_context(top_session_ids, compressed_sessions)
        
        # Генерируем fallback промпт
        prompt = get_fallback_prompt(question, memory_text)
        
        # Метаданные для отладки
        metadata = {
            'strategy': 'compressed_fallback',
            'topic': None,
            'confidence': 0.0,
            'total_sessions': len(compressed_sessions),
            'relevant_sessions': len(compressed_sessions),
            'selected_sessions': len(top_session_ids),
            'selected_session_ids': top_session_ids,
            'memory_length': len(memory_text),
            'compression_enabled': self.config.enable_compression
        }
        
        return prompt, metadata
    
    def _find_relevant_compressed_sessions(self, question: str, topic: str, 
                                         compressed_sessions: Dict[str, str]) -> List[str]:
        """Находит релевантные сессии среди сжатых"""
        relevant_sessions = []
        
        # Извлекаем ключевые слова из вопроса
        question_words = set(question.lower().split())
        
        for session_id, compressed_text in compressed_sessions.items():
            # Простой keyword matching
            text_words = set(compressed_text.lower().split())
            
            # Вычисляем пересечение слов
            common_words = question_words.intersection(text_words)
            
            # Если есть пересечение или текст содержит тему
            if common_words or (topic and topic.lower() in compressed_text.lower()):
                relevant_sessions.append(session_id)
        
        return relevant_sessions
    
    def _rank_compressed_sessions(self, question: str, session_ids: List[str],
                                compressed_sessions: Dict[str, str]) -> List[str]:
        """Ранжирует сжатые сессии по релевантности к вопросу"""
        if not session_ids:
            return []
        
        # Простое ранжирование по количеству общих слов
        question_words = set(question.lower().split())
        
        scored_sessions = []
        for session_id in session_ids:
            if session_id in compressed_sessions:
                text = compressed_sessions[session_id]
                text_words = set(text.lower().split())
                
                # Количество общих слов
                common_words = len(question_words.intersection(text_words))
                
                # Бонус за длину (более информативные сессии)
                length_bonus = min(len(text.split()) / 100, 1.0)
                
                score = common_words + length_bonus
                scored_sessions.append((session_id, score))
        
        # Сортируем по убыванию релевантности
        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        
        return [session_id for session_id, _ in scored_sessions]
    
    def _build_compressed_memory_context(self, session_ids: List[str], 
                                       compressed_sessions: Dict[str, str]) -> str:
        """Строит контекст памяти из сжатых сессий"""
        if not session_ids:
            return ""
        
        context_parts = []
        
        for session_id in session_ids:
            if session_id in compressed_sessions:
                compressed_text = compressed_sessions[session_id]
                if compressed_text.strip():
                    session_marker = f"[Сессия {session_id}]"
                    context_parts.append(f"{session_marker} {compressed_text}")
        
        return "\n\n".join(context_parts)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сжатия"""
        stats = {
            'total_dialogues': len(self.compressed_sessions),
            'total_sessions': sum(len(sessions) for sessions in self.compressed_sessions.values()),
            'compression_enabled': self.config.enable_compression,
            'compression_level': self.config.compression_level.value,
            'compression_method': self.config.compression_method.value,
            'hierarchical_enabled': self.config.enable_hierarchical_compression
        }
        
        # Добавляем статистику компрессора
        if hasattr(self.compressor, 'get_stats'):
            compressor_stats = self.compressor.get_stats()
            stats['compressor_stats'] = compressor_stats.to_dict()
        
        return stats
    
    def clear_compression_cache(self):
        """Очищает кэш сжатых сессий"""
        self.compressed_sessions.clear()
        logger.info("Кэш сжатых сессий очищен")
    
    def save_compressed_sessions(self, save_dir: str):
        """Сохраняет сжатые сессии на диск"""
        import json
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        for dialogue_id, sessions in self.compressed_sessions.items():
            filepath = save_path / f"compressed_sessions_{dialogue_id}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Сжатые сессии сохранены в {save_dir}")
    
    def load_compressed_sessions(self, save_dir: str):
        """Загружает сжатые сессии с диска"""
        import json
        from pathlib import Path
        
        save_path = Path(save_dir)
        if not save_path.exists():
            logger.warning(f"Директория {save_dir} не существует")
            return
        
        for filepath in save_path.glob("compressed_sessions_*.json"):
            dialogue_id = filepath.stem.replace("compressed_sessions_", "")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            self.compressed_sessions[dialogue_id] = sessions
        
        logger.info(f"Сжатые сессии загружены из {save_dir}")
