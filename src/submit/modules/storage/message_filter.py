"""
Центральный модуль для фильтрации сообщений от копипаста и технического контента.
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ с улучшенным определением личной информации
"""
from typing import List, Dict, Tuple, Set
from models import Message, Dialog
import re
from functools import lru_cache

from .filters.message_cleaner import (
    is_copy_paste_content,
    is_personal_message,
    is_technical_content,
    get_message_quality_score
)


class MessageFilter:
    """Центральный класс для фильтрации сообщений с оптимизацией"""
    
    def __init__(self):
        """Инициализация фильтра сообщений"""
        # Расширенный список личных маркеров с весами
        self.weighted_personal_markers = {
            # Сильные личные маркеры (вес 1.0)
            'я': 1.0, 'меня': 1.0, 'мне': 1.0, 'мной': 1.0, 'мною': 1.0,
            'мой': 1.0, 'моя': 1.0, 'моё': 1.0, 'мое': 1.0, 'мои': 1.0,
            
            # Средние личные маркеры (вес 0.7)
            'мы': 0.7, 'нас': 0.7, 'нам': 0.7, 'нами': 0.7,
            'наш': 0.7, 'наша': 0.7, 'наше': 0.7, 'наши': 0.7,
            
            # Слабые личные маркеры (вес 0.5)
            'свой': 0.5, 'своя': 0.5, 'своё': 0.5, 'свое': 0.5, 'свои': 0.5,
            'сам': 0.5, 'сама': 0.5, 'само': 0.5, 'сами': 0.5,
            
            # Контекстные личные слова (вес 0.3)
            'люблю': 0.3, 'хочу': 0.3, 'думаю': 0.3, 'считаю': 0.3,
            'нравится': 0.3, 'интересно': 0.3, 'важно': 0.3,
        }
        
        # Контекстные фразы, которые делают копипаст личным
        self.personal_copypaste_phrases = {
            'помоги мне', 'моя задача', 'мой вопрос', 'я хочу', 
            'мне нужно', 'я не понимаю', 'объясни мне', 'расскажи мне',
            'мой проект', 'моя работа', 'мое задание', 'для меня'
        }
        
        # Кэш для результатов анализа
        self._analysis_cache = {}
        self._cache_size = 500
    
    @lru_cache(maxsize=1000)
    def _calculate_personal_score(self, content: str) -> float:
        """
        Рассчитывает персональный скор сообщения с учетом весов
        
        Args:
            content: Содержимое сообщения
            
        Returns:
            Скор от 0.0 до 1.0
        """
        if not content:
            return 0.0
        
        content_lower = content.lower()
        words = re.findall(r'\b[а-яёa-z]+\b', content_lower)
        
        if not words:
            return 0.0
        
        total_score = 0.0
        personal_words_count = 0
        
        # Подсчитываем взвешенный скор
        for word in words:
            if word in self.weighted_personal_markers:
                weight = self.weighted_personal_markers[word]
                total_score += weight
                personal_words_count += 1
        
        # Проверяем личные фразы
        for phrase in self.personal_copypaste_phrases:
            if phrase in content_lower:
                total_score += 2.0  # Фразы имеют больший вес
        
        # Нормализуем скор
        if personal_words_count > 0:
            # Учитываем плотность личных слов
            density = personal_words_count / len(words)
            final_score = (total_score / len(words)) * (1 + density)
            return min(1.0, final_score)
        
        return 0.0
    
    def _is_personal_copypaste(self, content: str) -> bool:
        """
        Определяет, является ли копипаст личным (содержит личный контекст)
        
        Args:
            content: Содержимое сообщения
            
        Returns:
            True если копипаст содержит личную информацию
        """
        content_lower = content.lower()
        
        # Проверяем личные фразы в копипасте
        for phrase in self.personal_copypaste_phrases:
            if phrase in content_lower:
                return True
        
        # Проверяем персональный скор
        personal_score = self._calculate_personal_score(content)
        
        # Если скор высокий, то даже копипаст считаем личным
        return personal_score > 0.3
    
    def filter_dialogs(self, dialogs: List[Dialog]) -> List[Dialog]:
        """
        Фильтрует диалоги с оптимизацией
        """
        filtered_dialogs = []
        
        for dialog in dialogs:
            # Используем контекст диалога для лучшей фильтрации
            dialog_context = self._build_dialog_context(dialog)
            
            filtered_sessions = []
            for session in dialog.sessions:
                filtered_messages = self._filter_with_context(
                    session.messages, 
                    dialog_context
                )
                
                if filtered_messages:
                    from models import Session
                    filtered_session = Session(
                        id=session.id,
                        messages=filtered_messages
                    )
                    filtered_sessions.append(filtered_session)
            
            if filtered_sessions:
                filtered_dialog = Dialog(
                    id=dialog.id,
                    question=dialog.question,
                    sessions=filtered_sessions
                )
                filtered_dialogs.append(filtered_dialog)
        
        return filtered_dialogs
    
    def filter_messages(self, messages: List[Message]) -> List[Message]:
        """
        Фильтрует сообщения с оптимизированным определением личной информации
        """
        if not messages:
            return []
        
        # Строим контекст из всех сообщений
        context = self._build_message_context(messages)
        
        # Фильтруем с учетом контекста
        return self._filter_with_context(messages, context)
    
    def _filter_with_context(self, messages: List[Message], context: Dict) -> List[Message]:
        """
        Фильтрация с учетом контекста диалога
        """
        filtered = []
        previous_was_copypaste = False
        personal_streak = 0  # Счетчик последовательных личных сообщений
        
        for i, msg in enumerate(messages):
            if msg.role == "user" and msg.content.strip():
                # Проверяем кэш
                cache_key = hash(msg.content[:100])
                if cache_key in self._analysis_cache:
                    analysis = self._analysis_cache[cache_key]
                else:
                    # Анализируем сообщение
                    analysis = self._analyze_message(msg.content, context)
                    
                    # Кэшируем результат
                    if len(self._analysis_cache) < self._cache_size:
                        self._analysis_cache[cache_key] = analysis
                
                # Решение о включении сообщения
                should_include = False
                
                if analysis['is_personal']:
                    # Личное сообщение - включаем
                    should_include = True
                    personal_streak += 1
                elif analysis['is_personal_copypaste']:
                    # Копипаст с личным контекстом - включаем
                    should_include = True
                    personal_streak = 0
                elif personal_streak >= 2 and not analysis['is_pure_copypaste']:
                    # После 2+ личных сообщений более лояльны к следующим
                    should_include = True
                    personal_streak += 1
                else:
                    personal_streak = 0
                
                # Дополнительная проверка на предыдущий копипаст
                if previous_was_copypaste and analysis['is_copypaste']:
                    should_include = False
                
                if should_include:
                    filtered.append(msg)
                    previous_was_copypaste = False
                else:
                    previous_was_copypaste = analysis['is_copypaste']
                    
            elif msg.role == "assistant":
                # Сохраняем ответы ассистента если есть контекст
                if filtered or personal_streak > 0:
                    filtered.append(msg)
                previous_was_copypaste = False
        
        return filtered
    
    def _analyze_message(self, content: str, context: Dict) -> Dict:
        """
        Полный анализ сообщения с учетом контекста
        """
        personal_score = self._calculate_personal_score(content)
        is_copypaste = is_copy_paste_content(content)
        is_technical = is_technical_content(content)
        
        # Определяем тип сообщения
        is_personal = personal_score > 0.2 and not is_copypaste
        is_personal_copypaste = is_copypaste and self._is_personal_copypaste(content)
        is_pure_copypaste = is_copypaste and not is_personal_copypaste
        
        # Учитываем контекст диалога
        if context.get('has_strong_personal_context'):
            # В личном контексте снижаем порог
            is_personal = personal_score > 0.1
        
        return {
            'personal_score': personal_score,
            'is_copypaste': is_copypaste,
            'is_technical': is_technical,
            'is_personal': is_personal,
            'is_personal_copypaste': is_personal_copypaste,
            'is_pure_copypaste': is_pure_copypaste
        }
    
    def _build_dialog_context(self, dialog: Dialog) -> Dict:
        """
        Строит контекст диалога для улучшенной фильтрации
        """
        context = {
            'has_strong_personal_context': False,
            'main_topics': set(),
            'personal_words_count': 0
        }
        
        # Анализируем вопрос диалога
        if dialog.question:
            question_score = self._calculate_personal_score(dialog.question)
            if question_score > 0.3:
                context['has_strong_personal_context'] = True
        
        # Быстрый анализ первых сообщений для определения контекста
        for session in dialog.sessions[:2]:  # Смотрим первые 2 сессии
            for msg in session.messages[:5]:  # Первые 5 сообщений
                if msg.role == "user":
                    score = self._calculate_personal_score(msg.content)
                    if score > 0.3:
                        context['personal_words_count'] += 1
        
        if context['personal_words_count'] >= 2:
            context['has_strong_personal_context'] = True
        
        return context
    
    def _build_message_context(self, messages: List[Message]) -> Dict:
        """
        Строит контекст из списка сообщений
        """
        context = {
            'has_strong_personal_context': False,
            'personal_messages_ratio': 0.0
        }
        
        if not messages:
            return context
        
        # Считаем личные сообщения
        personal_count = 0
        user_count = 0
        
        for msg in messages[:10]:  # Анализируем первые 10 сообщений
            if msg.role == "user":
                user_count += 1
                score = self._calculate_personal_score(msg.content)
                if score > 0.3:
                    personal_count += 1
        
        if user_count > 0:
            context['personal_messages_ratio'] = personal_count / user_count
            if context['personal_messages_ratio'] > 0.5:
                context['has_strong_personal_context'] = True
        
        return context
    
    def filter_messages_simple(self, messages: List[Message]) -> List[Message]:
        """
        Простая фильтрация сообщений (только удаление копипаста)
        """
        if not messages:
            return []
        
        filtered_messages = []
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                # Простая проверка без контекста
                if not is_copy_paste_content(msg.content) or \
                   self._is_personal_copypaste(msg.content):
                    filtered_messages.append(msg)
            else:
                filtered_messages.append(msg)
        
        return filtered_messages
    
    def get_message_analysis(self, messages: List[Message]) -> Dict[str, any]:
        """
        Анализирует сообщения и возвращает статистику фильтрации
        """
        if not messages:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'filtered_messages': 0,
                'copypaste_messages': 0,
                'technical_messages': 0,
                'personal_messages': 0,
                'personal_copypaste_messages': 0,
                'filter_ratio': 0.0,
                'avg_personal_score': 0.0
            }
        
        total_messages = len(messages)
        user_messages = sum(1 for msg in messages if msg.role == "user")
        
        copypaste_count = 0
        technical_count = 0
        personal_count = 0
        personal_copypaste_count = 0
        total_personal_score = 0.0
        
        context = self._build_message_context(messages)
        
        for msg in messages:
            if msg.role == "user" and msg.content.strip():
                analysis = self._analyze_message(msg.content, context)
                
                if analysis['is_copypaste']:
                    copypaste_count += 1
                if analysis['is_technical']:
                    technical_count += 1
                if analysis['is_personal']:
                    personal_count += 1
                if analysis['is_personal_copypaste']:
                    personal_copypaste_count += 1
                
                total_personal_score += analysis['personal_score']
        
        filtered_messages = self.filter_messages(messages)
        filtered_count = len(filtered_messages)
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'filtered_messages': filtered_count,
            'copypaste_messages': copypaste_count,
            'technical_messages': technical_count,
            'personal_messages': personal_count,
            'personal_copypaste_messages': personal_copypaste_count,
            'filter_ratio': (user_messages - filtered_count) / user_messages if user_messages > 0 else 0.0,
            'avg_personal_score': total_personal_score / user_messages if user_messages > 0 else 0.0
        }
    
    def get_message_quality_analysis(self, messages: List[Message]) -> List[Dict[str, any]]:
        """
        Анализирует качество каждого сообщения с детальным анализом
        """
        analysis = []
        context = self._build_message_context(messages)
        
        for i, msg in enumerate(messages):
            if msg.role == "user" and msg.content.strip():
                detailed_analysis = self._analyze_message(msg.content, context)
                
                analysis.append({
                    'index': i,
                    'content_preview': msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    'content_length': len(msg.content),
                    'personal_score': detailed_analysis['personal_score'],
                    'quality_score': get_message_quality_score(msg.content),
                    'is_copypaste': detailed_analysis['is_copypaste'],
                    'is_personal_copypaste': detailed_analysis['is_personal_copypaste'],
                    'is_technical': detailed_analysis['is_technical'],
                    'is_personal': detailed_analysis['is_personal'],
                    'will_be_filtered': detailed_analysis['is_pure_copypaste'] and not detailed_analysis['is_personal']
                })
        
        return analysis


# Глобальный экземпляр для удобства использования
message_filter = MessageFilter()


def filter_dialogs(dialogs: List[Dialog]) -> List[Dialog]:
    """
    Простая функция для фильтрации диалогов
    (для обратной совместимости)
    """
    return message_filter.filter_dialogs(dialogs)


def filter_messages(messages: List[Message]) -> List[Message]:
    """
    Простая функция для фильтрации сообщений
    (для обратной совместимости)
    """
    return message_filter.filter_messages(messages)