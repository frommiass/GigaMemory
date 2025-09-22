"""
Модуль для очистки сообщений от мусорного контента
"""
import re
from typing import List, Set
from models import Message

from .regex_patterns import (
    COMPILED_PATTERNS,
    PERSONAL_MARKERS,
    COPYPASTE_MARKERS,
    TECH_SIGNS
)


def _check_markers(text: str, markers: Set[str]) -> bool:
    """
    Проверяет наличие маркеров как целых слов в тексте
    """
    words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]+\b', text.lower())
    words_set = set(words)
    return bool(words_set & markers)


def is_personal_message(content: str) -> bool:
    """
    Определяет, содержит ли сообщение личную информацию
    """
    # Быстрые проверки по длине
    if len(content) > 500:
        return False
    
    if len(content) < 100:
        return True
    
    content_lower = content.lower()
    
    # Проверка личных местоимений
    if _check_markers(content_lower, PERSONAL_MARKERS):
        return True
    
    # Проверка маркеров копипаста
    if _check_markers(content_lower, COPYPASTE_MARKERS):
        return False
    
    # Технические признаки
    if _check_markers(content, TECH_SIGNS):
        return False
    
    # Подсчет статистики в один проход
    digits_count = 0
    punct_count = 0
    caps_sequences = 0
    prev_was_upper = False
    newlines_count = 0
    
    for char in content:
        if char.isdigit():
            digits_count += 1
        elif char in '.,;:!?()[]{}/"\'':
            punct_count += 1
        elif char == '\n':
            newlines_count += 1
        elif char.isupper():
            if prev_was_upper:
                caps_sequences += 1
            prev_was_upper = True
        else:
            prev_was_upper = False
    
    # Пороговые проверки
    content_len = len(content)
    if digits_count > content_len * 0.25:
        return False
    if punct_count > content_len * 0.3:
        return False
    if caps_sequences > 20:
        return False
    if newlines_count > 10:
        return False
    
    # Проверка уникальности слов
    words = content.split()
    if len(words) > 20:
        unique_words = len(set(words))
        if unique_words / len(words) < 0.5:
            return False
    
    # Проверка регулярными выражениями
    for pattern in COMPILED_PATTERNS:
        if pattern.search(content):
            return False
    
    # Средняя длина предложений
    if content_len > 200:
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) > 3:
            total_length = sum(len(s) for s in sentences)
            avg_sentence_length = total_length / len(sentences)
            if avg_sentence_length > 150:
                return False
    
    return True


def clean_messages(messages: List[Message]) -> List[Message]:
    """
    Очищает сообщения от мусорного контента, оставляя только личную информацию
    
    Args:
        messages: Список сообщений для очистки
        
    Returns:
        Список очищенных сообщений пользователя
    """
    cleaned_messages = []
    
    for msg in messages:
        if msg.role == "user":
            if is_personal_message(msg.content):
                cleaned_messages.append(msg)
    
    return cleaned_messages


def clean_user_messages(messages: List[Message]) -> List[Message]:
    """
    Алиас для обратной совместимости с существующим кодом
    """
    return clean_messages(messages)


def is_technical_content(content: str) -> bool:
    """
    Проверяет, является ли контент техническим (код, конфиги и т.д.)
    
    Args:
        content: Содержимое сообщения
        
    Returns:
        True если контент технический
    """
    content_lower = content.lower()
    
    # Проверка технических маркеров
    if _check_markers(content, TECH_SIGNS):
        return True
    
    # Проверка маркеров копипаста
    if _check_markers(content_lower, COPYPASTE_MARKERS):
        return True
    
    # Проверка регулярными выражениями
    for pattern in COMPILED_PATTERNS:
        if pattern.search(content):
            return True
    
    return False


def is_copy_paste_content(content: str) -> bool:
    """
    Проверяет, является ли контент копипастом
    
    Args:
        content: Содержимое сообщения
        
    Returns:
        True если контент копипаст
    """
    content_lower = content.lower()
    return _check_markers(content_lower, COPYPASTE_MARKERS)


def get_message_quality_score(content: str) -> float:
    """
    Рассчитывает качество сообщения (0.0 - 1.0)
    
    Args:
        content: Содержимое сообщения
        
    Returns:
        Оценка качества от 0.0 до 1.0
    """
    if not content or len(content) < 10:
        return 0.0
    
    score = 1.0
    
    # Штраф за технический контент
    if is_technical_content(content):
        score -= 0.5
    
    # Штраф за копипаст
    if is_copy_paste_content(content):
        score -= 0.3
    
    # Штраф за слишком длинные сообщения
    if len(content) > 1000:
        score -= 0.2
    
    # Бонус за личные местоимения
    content_lower = content.lower()
    if _check_markers(content_lower, PERSONAL_MARKERS):
        score += 0.2
    
    # Штраф за избыток цифр
    digits_count = sum(1 for c in content if c.isdigit())
    if digits_count > len(content) * 0.3:
        score -= 0.3
    
    return max(0.0, min(1.0, score))
