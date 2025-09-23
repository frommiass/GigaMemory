"""
Оптимизированная версия model_inference.py для соревнования GigaMemory
С учетом фильтрации копипаста и обработки больших диалогов
"""
from typing import List, Dict, Optional, Any
from models import Message
from submit_interface import ModelWithMemory

from .storage import MemoryStorage
from .llm_inference import ModelInference
from .smart_memory import SmartMemory, SmartMemoryConfig
from .compression import CompressionLevel


class SubmitModelWithMemory(ModelWithMemory):
    """
    Оптимизированная система памяти для соревнования
    - Фильтрация копипаста
    - Только сообщения пользователей
    - Оптимизация для больших диалогов (100к+ символов)
    """

    def __init__(self, model_path: str) -> None:
        """
        Инициализация с оптимальными настройками для соревнования
        """
        # Базовые компоненты
        self.storage = MemoryStorage()
        self.model_inference = ModelInference(model_path)
        
        # Настройка SmartMemory с оптимизациями
        config = SmartMemoryConfig()
        
        # Векторный поиск - обязательно для больших диалогов
        config.use_vector_search = True
        config.embedding_model = "cointegrated/rubert-tiny2"  # Легкая русская модель
        config.vector_top_k = 5  # Топ-5 релевантных сессий
        
        # Извлечение фактов - важно для соревнования
        config.use_fact_extraction = True
        config.fact_min_confidence = 0.6  # Средний порог для баланса точности/полноты
        
        # Сжатие - критично для больших диалогов
        config.use_compression = True
        config.compression_level = CompressionLevel.MODERATE  # Умеренное сжатие
        
        # Гибридный поиск для лучших результатов
        config.use_hybrid_search = True
        config.max_context_length = 2048  # Оптимально для GigaChat
        
        # Инициализируем SmartMemory
        self.smart_memory = SmartMemory(model_path, config)
        
        # Кэш для фильтрации (критично для производительности)
        self.filter_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Статистика обработки
        self.stats = {
            'total_messages': 0,
            'filtered_messages': 0,
            'copypaste_filtered': 0,
            'sessions_created': 0,
            'facts_extracted': 0,
            'compression_ratio': 1.0
        }

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        """
        Фильтрует копипаст и сохраняет только личную информацию пользователя
        Оптимизировано для больших диалогов
        """
        from .core.message_filter import message_filter
        from .filters.session_grouper import SessionGrouper
        
        # Обновляем статистику
        self.stats['total_messages'] += len(messages)
        
        # 1. Фильтруем сообщения (ТОЛЬКО пользователя, БЕЗ копипаста)
        filtered_messages = []
        
        for msg in messages:
            # ВАЖНО: Обрабатываем ТОЛЬКО сообщения пользователя
            if msg.role != "user":
                continue
            
            # Проверяем кэш для ускорения
            content_hash = hash(msg.content[:100])  # Хэш первых 100 символов
            
            if content_hash in self.filter_cache:
                should_keep = self.filter_cache[content_hash]
                self.cache_hits += 1
            else:
                # Проверяем на копипаст и личную информацию
                from .filters.message_cleaner import is_copy_paste_content, is_personal_message
                
                # Фильтруем копипаст
                is_copypaste = is_copy_paste_content(msg.content)
                if is_copypaste:
                    self.stats['copypaste_filtered'] += 1
                    should_keep = False
                else:
                    # Проверяем на личную информацию
                    should_keep = is_personal_message(msg.content)
                
                # Сохраняем в кэш
                self.filter_cache[content_hash] = should_keep
                self.cache_misses += 1
                
                # Ограничиваем размер кэша
                if len(self.filter_cache) > 10000:
                    # Удаляем первые 1000 элементов
                    items = list(self.filter_cache.items())
                    self.filter_cache = dict(items[1000:])
            
            if should_keep:
                filtered_messages.append(msg)
                self.stats['filtered_messages'] += 1
        
        # Если нет отфильтрованных сообщений, выходим
        if not filtered_messages:
            return
        
        # 2. Группируем по сессиям
        grouper = SessionGrouper()
        sessions = {}
        
        for msg in filtered_messages:
            # Используем session_id из сообщения или создаем новый
            session_id = msg.session_id if msg.session_id else str(self.storage.increment_session(dialogue_id))
            
            if session_id not in sessions:
                sessions[session_id] = []
                self.stats['sessions_created'] += 1
            
            sessions[session_id].append(msg)
        
        # 3. Сохраняем в память с оптимизациями
        for session_id, session_messages in sessions.items():
            # Батчевая обработка для эффективности
            batch = []
            for msg in session_messages:
                processed_msg = Message(
                    role=msg.role,
                    content=msg.content.strip(),  # Убираем лишние пробелы
                    session_id=session_id
                )
                batch.append(processed_msg)
            
            # Добавляем батч в память
            self.storage.add_to_memory(dialogue_id, batch)
        
        # 4. Обрабатываем через SmartMemory для извлечения фактов и векторизации
        if sessions:
            # Преобразуем в формат для SmartMemory
            all_messages = []
            for session_messages in sessions.values():
                all_messages.extend(session_messages)
            
            # Обрабатываем диалог (извлечение фактов, векторизация, сжатие)
            processing_stats = self.smart_memory.process_dialogue(dialogue_id, all_messages)
            
            # Обновляем статистику
            self.stats['facts_extracted'] += processing_stats.get('facts_extracted', 0)
            self.stats['compression_ratio'] = processing_stats.get('compression_ratio', 1.0)

    def clear_memory(self, dialogue_id: str) -> None:
        """
        Очищает память диалога и оптимизирует кэши
        """
        # Очищаем основную память
        self.storage.clear_dialogue_memory(dialogue_id)
        
        # Очищаем кэш фильтрации если слишком большой
        if len(self.filter_cache) > 5000:
            self.filter_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        
        # Очищаем кэш хранилища если нужно
        if self.storage.get_cache_size() > 1000:
            self.storage.clear_all_cache()

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        """
        Генерирует ответ на вопрос используя все компоненты системы
        Оптимизировано для точных ответов
        """
        try:
            # Получаем память диалога
            memory = self.storage.get_memory(dialogue_id)
            
            if not memory:
                # Если памяти нет, используем дефолтный ответ
                return self._get_no_info_answer(question)
            
            # Используем SmartMemory для генерации ответа
            # (векторный поиск + факты + сжатие)
            answer = self.smart_memory.answer_question(dialogue_id, question)
            
            # Постобработка ответа
            answer = self._postprocess_answer(answer, question)
            
            return answer
            
        except Exception as e:
            # Логируем ошибку но не падаем
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка при ответе на вопрос: {e}")
            
            # Возвращаем fallback ответ
            return self._get_fallback_answer(question)

    def _get_no_info_answer(self, question: str) -> str:
        """Возвращает ответ когда нет информации"""
        question_lower = question.lower()
        
        # Специфичные ответы для типовых вопросов
        if "имя" in question_lower or "зовут" in question_lower:
            return "Я не знаю вашего имени."
        elif "возраст" in question_lower or "лет" in question_lower:
            return "Я не знаю вашего возраста."
        elif "живете" in question_lower or "город" in question_lower:
            return "Я не знаю, где вы живете."
        elif "работа" in question_lower or "профессия" in question_lower:
            return "Я не знаю, где вы работаете."
        else:
            return "У меня нет информации для ответа на этот вопрос."

    def _get_fallback_answer(self, question: str) -> str:
        """Возвращает fallback ответ при ошибке"""
        return "К сожалению, я не могу ответить на этот вопрос."

    def _postprocess_answer(self, answer: str, question: str) -> str:
        """
        Постобработка ответа для улучшения качества
        """
        # Убираем лишние пробелы
        answer = answer.strip()
        
        # Ограничиваем длину ответа (требование соревнования - короткие ответы)
        if len(answer) > 100:
            # Обрезаем до первого предложения
            first_sentence = answer.split('.')[0]
            if first_sentence:
                answer = first_sentence + '.'
        
        # Проверяем, что ответ не пустой
        if not answer or answer == ".":
            answer = self._get_no_info_answer(question)
        
        # Убираем повторения
        words = answer.split()
        if len(words) > 3:
            # Проверяем на повторяющиеся слова
            unique_words = []
            prev_word = ""
            for word in words:
                if word.lower() != prev_word.lower():
                    unique_words.append(word)
                prev_word = word
            answer = " ".join(unique_words)
        
        return answer

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы системы"""
        cache_hit_rate = 0.0
        if self.cache_hits + self.cache_misses > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
        
        return {
            'processing_stats': self.stats,
            'cache_stats': {
                'cache_size': len(self.filter_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': cache_hit_rate
            },
            'memory_stats': {
                'total_dialogues': len(self.storage.dialogue_memory),
                'total_sessions': sum(
                    len(msgs) for msgs in self.storage.dialogue_memory.values()
                )
            },
            'smart_memory_stats': self.smart_memory.get_stats()
        }

    def optimize_for_large_dialogue(self):
        """Оптимизация для больших диалогов (100к+ символов)"""
        # Увеличиваем размер кэша
        self.filter_cache.clear()
        
        # Настраиваем более агрессивное сжатие
        self.smart_memory.config.compression_level = CompressionLevel.AGGRESSIVE
        
        # Увеличиваем батчи для векторизации
        if hasattr(self.smart_memory, 'rag_engine'):
            self.smart_memory.rag_engine.config.embedding_batch_size = 64
        
        # Оптимизируем память
        import gc
        gc.collect()
        
        return "Система оптимизирована для больших диалогов"
