# src/submit/modules/storage/module.py

from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../..', 'memory_aij2025'))
from models import Message
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'submit'))
from core.interfaces import IStorage, ProcessingResult

from .memory_storage import MemoryStorage
from .message_filter import MessageFilter
from .filters.session_grouper import SessionGrouper
from .filters.relevance_filter import RelevanceFilter


class StorageModule(IStorage):
    """Модуль управления хранилищем и сессиями"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory = MemoryStorage()
        self.filter = MessageFilter()
        self.grouper = SessionGrouper()
        self.relevance = RelevanceFilter(config)
        self.optimizer = None  # Будет установлен через set_dependencies
        
        # Кэш для результатов фильтрации
        self._filter_cache = {}
        self._cache_max_size = 1000
    
    # === КРИТИЧНО: Исправленные методы интерфейса IStorage ===
    
    def store_messages(self, dialogue_id: str, messages: List[Message]) -> ProcessingResult:
        """
        Сохраняет сообщения с фильтрацией (переименован из add_messages)
        
        Args:
            dialogue_id: ID диалога
            messages: Список сообщений для сохранения
            
        Returns:
            ProcessingResult с отфильтрованными сообщениями
        """
        try:
            # Проверяем кэш фильтрации
            cache_key = f"{dialogue_id}_{len(messages)}_{hash(str(messages[:3]))}"
            
            if cache_key in self._filter_cache:
                filtered = self._filter_cache[cache_key]
            else:
                # Фильтруем сообщения с оптимизацией
                filtered = self._optimized_filter(messages)
                
                # Кэшируем результат
                if len(self._filter_cache) < self._cache_max_size:
                    self._filter_cache[cache_key] = filtered
                else:
                    # Очищаем половину кэша при переполнении
                    items = list(self._filter_cache.items())
                    self._filter_cache = dict(items[len(items)//2:])
                    self._filter_cache[cache_key] = filtered
            
            # Группируем по сессиям
            sessions = self.grouper.group_messages_by_sessions(filtered, dialogue_id)
            
            # Сохраняем в память
            self.memory.add_to_memory(dialogue_id, filtered)
            
            # Если есть optimizer, оптимизируем хранение
            if self.optimizer:
                self.optimizer.optimize_storage(dialogue_id, filtered)
            
            return ProcessingResult(
                success=True,
                data=filtered,
                metadata={
                    'total': len(messages),
                    'filtered': len(filtered),
                    'sessions': len(sessions),
                    'filter_efficiency': 1 - (len(filtered) / len(messages)) if messages else 0
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    def get_dialogue_messages(self, dialogue_id: str) -> ProcessingResult:
        """
        Получает все сообщения диалога (переименован из get_messages)
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            ProcessingResult со всеми сообщениями диалога
        """
        try:
            messages = self.memory.get_memory(dialogue_id)
            
            # Если есть optimizer, используем оптимизированное получение
            if self.optimizer:
                messages = self.optimizer.get_optimized_messages(dialogue_id, messages)
            
            return ProcessingResult(
                success=True,
                data=messages,
                metadata={
                    'count': len(messages),
                    'dialogue_id': dialogue_id
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )
    
    def get_dialogue_sessions(self, dialogue_id: str) -> ProcessingResult:
        """
        НОВЫЙ МЕТОД: Получает сессии диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            ProcessingResult со словарем сессий {session_id: [messages]}
        """
        try:
            # Получаем все сообщения
            messages = self.memory.get_memory(dialogue_id)
            
            if not messages:
                return ProcessingResult(
                    success=True,
                    data={},
                    metadata={
                        'dialogue_id': dialogue_id,
                        'sessions_count': 0,
                        'messages_count': 0
                    }
                )
            
            # Группируем по сессиям
            sessions = self.grouper.group_messages_by_sessions(messages, dialogue_id)
            
            # Получаем дополнительную информацию о сессиях
            session_info = {}
            for session_id, session_messages in sessions.items():
                from .filters.session_grouper import get_session_summary
                session_info[session_id] = get_session_summary(session_messages)
            
            return ProcessingResult(
                success=True,
                data=sessions,
                metadata={
                    'dialogue_id': dialogue_id,
                    'sessions_count': len(sessions),
                    'messages_count': len(messages),
                    'session_info': session_info
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def set_dependencies(self, optimizer=None):
        """
        НОВЫЙ МЕТОД: Устанавливает зависимости от других модулей
        
        Args:
            optimizer: Экземпляр модуля оптимизации (опционально)
        """
        self.optimizer = optimizer
        
        # Если optimizer предоставлен, настраиваем дополнительные параметры
        if self.optimizer:
            # Увеличиваем размер кэша при наличии оптимизатора
            self._cache_max_size = 2000
            
            # Передаем конфигурацию оптимизатору
            if hasattr(self.optimizer, 'configure_for_storage'):
                self.optimizer.configure_for_storage(self.config)
    
    def clear(self, dialogue_id: str) -> ProcessingResult:
        """Очищает данные диалога"""
        try:
            self.memory.clear_dialogue_memory(dialogue_id)
            self.grouper.clear_dialogue_sessions(dialogue_id)
            
            # Очищаем кэш для этого диалога
            keys_to_remove = [k for k in self._filter_cache if k.startswith(dialogue_id)]
            for key in keys_to_remove:
                del self._filter_cache[key]
            
            return ProcessingResult(
                success=True,
                data=None,
                metadata={'cleared': True}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=None,
                metadata={},
                error=str(e)
            )
    
    # === Оптимизированные методы фильтрации ===
    
    def _optimized_filter(self, messages: List[Message]) -> List[Message]:
        """
        Оптимизированная фильтрация с улучшенным определением личной информации
        
        Args:
            messages: Список сообщений для фильтрации
            
        Returns:
            Отфильтрованные сообщения
        """
        if not messages:
            return []
        
        from .filters.message_cleaner import is_copy_paste_content, is_personal_message
        from .filters.regex_patterns import PERSONAL_MARKERS
        
        filtered = []
        previous_was_copypaste = False
        personal_context_active = False
        
        for i, msg in enumerate(messages):
            if msg.role == "user" and msg.content.strip():
                content_lower = msg.content.lower()
                
                # Улучшенное определение личной информации
                # 1. Проверяем личные маркеры
                has_personal_markers = any(marker in content_lower for marker in PERSONAL_MARKERS)
                
                # 2. Проверяем контекст (если предыдущие сообщения были личными)
                if i > 0 and personal_context_active:
                    # В личном контексте более лояльно относимся к сообщениям
                    has_personal_context = True
                else:
                    has_personal_context = False
                
                # 3. Проверяем на копипаст с учетом контекста
                is_copypaste = is_copy_paste_content(msg.content)
                
                # Если есть личные маркеры, то даже копипаст может быть личным
                # (например, "помоги мне с моей задачей")
                if has_personal_markers and not previous_was_copypaste:
                    is_copypaste = False
                
                # 4. Решение о включении сообщения
                if has_personal_markers or has_personal_context:
                    if not is_copypaste:
                        filtered.append(msg)
                        personal_context_active = True
                        previous_was_copypaste = False
                    else:
                        previous_was_copypaste = True
                        personal_context_active = False
                elif is_personal_message(msg.content) and not is_copypaste:
                    filtered.append(msg)
                    personal_context_active = True
                    previous_was_copypaste = False
                else:
                    previous_was_copypaste = is_copypaste
                    personal_context_active = False
                    
            elif msg.role == "assistant":
                # Сохраняем ответы ассистента только если есть контекст
                if filtered or personal_context_active:
                    filtered.append(msg)
                previous_was_copypaste = False
        
        return filtered
    
    # === Методы для работы с вопросами и RAG (без изменений) ===
    
    def get_filtered_sessions_for_question(self, dialogue_id: str, question: str, 
                                          max_sessions: int = 5) -> ProcessingResult:
        """
        Получает отфильтрованные релевантные сессии для ответа на вопрос
        """
        try:
            # Получаем все сообщения
            messages = self.memory.get_memory(dialogue_id)
            
            # Группируем по сессиям
            all_sessions = self.grouper.group_messages_by_sessions(messages, dialogue_id)
            
            # Находим только релевантные сессии для вопроса
            relevant_sessions = self.relevance.find_relevant_sessions(
                question, all_sessions, max_sessions
            )
            
            # Дополнительная фильтрация контекста под вопрос
            filtered_sessions = {}
            for session_id, session_messages in relevant_sessions.items():
                filtered_msgs = self.filter_for_question_context(session_messages, question)
                if filtered_msgs:
                    filtered_sessions[session_id] = filtered_msgs
            
            return ProcessingResult(
                success=True,
                data=filtered_sessions,
                metadata={
                    'total_sessions': len(all_sessions),
                    'relevant_sessions': len(filtered_sessions),
                    'question': question
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def get_session_content_for_prompt(self, session_messages: List[Message]) -> str:
        """
        Извлекает чистый текст из сессии для промпта
        """
        from .filters.session_grouper import extract_session_content
        return extract_session_content(session_messages)
    
    def filter_for_question_context(self, messages: List[Message], question: str) -> List[Message]:
        """
        Специальная фильтрация сообщений для контекста вопроса
        """
        from .filters.message_cleaner import is_copy_paste_content, is_technical_content
        
        filtered = []
        question_words = set(question.lower().split())
        
        for msg in messages:
            # Проверяем кэш фильтрации
            cache_key = f"{msg.content[:50]}_{question[:20]}"
            cached = self.memory.check_cache(cache_key)
            
            if cached is not None:
                if cached:
                    filtered.append(msg)
                continue
            
            # Фильтруем
            should_include = True
            
            # Пропускаем если это явный копипаст без связи с вопросом
            if is_copy_paste_content(msg.content):
                content_words = set(msg.content.lower().split())
                # Но оставляем если есть пересечение с вопросом
                if not question_words & content_words:
                    should_include = False
            
            # Пропускаем технический контент без связи с вопросом
            if should_include and is_technical_content(msg.content):
                content_words = set(msg.content.lower().split())
                if not question_words & content_words:
                    should_include = False
            
            # Кэшируем результат
            self.memory.add_to_cache(cache_key, should_include)
            
            if should_include:
                filtered.append(msg)
        
        return filtered
    
    def get_relevance_scores(self, dialogue_id: str, question: str) -> ProcessingResult:
        """
        Получает оценки релевантности всех сессий для вопроса
        """
        try:
            # Получаем все сессии
            messages = self.memory.get_memory(dialogue_id)
            sessions = self.grouper.group_messages_by_sessions(messages, dialogue_id)
            
            # Получаем scores
            scores = self.relevance.get_relevance_scores(question, sessions)
            
            return ProcessingResult(
                success=True,
                data=scores,
                metadata={
                    'sessions_count': len(sessions),
                    'question': question
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def get_session_stats(self, dialogue_id: str) -> ProcessingResult:
        """
        Получает статистику по сессиям диалога
        """
        try:
            stats = self.grouper.get_session_stats(dialogue_id)
            return ProcessingResult(
                success=True,
                data=stats,
                metadata={'dialogue_id': dialogue_id}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def get_memory_stats(self, dialogue_id: str) -> ProcessingResult:
        """
        Получает статистику по памяти диалога
        """
        try:
            stats = self.memory.get_memory_stats(dialogue_id)
            
            # Добавляем информацию о кэше
            stats['filter_cache_size'] = len(self._filter_cache)
            stats['filter_cache_hits'] = self.memory.memory_registry.cache_hits
            stats['filter_cache_misses'] = self.memory.memory_registry.cache_misses
            
            return ProcessingResult(
                success=True,
                data=stats,
                metadata={'dialogue_id': dialogue_id}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )