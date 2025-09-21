"""
Модуль для фильтрации и обработки сообщений в GigaMemory
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
from .prompts import get_session_marker_prompt


def _check_markers(text: str, markers: Set[str]) -> bool:
    """
    Проверяет наличие маркеров как целых слов в тексте
    """
    words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]+\b', text.lower())
    words_set = set(words)
    return bool(words_set & markers)


def filter_user_message(content: str) -> bool:
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


def clean_user_messages(messages: List[Message]) -> List[Message]:
    """
    Оставляет только сообщения пользователя с личной информацией
    """
    filtered_messages = []
    
    for msg in messages:
        if msg.role == "user":
            if filter_user_message(msg.content):
                filtered_messages.append(msg)
        
    return filtered_messages


def process_sessions(messages: List[Message], session_id: int = 1) -> List[Message]:
    """
    Фильтрует сообщения и добавляет маркер сессии
    """
    clean_msgs = clean_user_messages(messages)
    
    with_sessions = []
    for msg in clean_msgs:
        user_msg = Message(
            role=msg.role,
            content=f"{get_session_marker_prompt(session_id)} {msg.content}"
        )
        with_sessions.append(user_msg)
    
    return with_sessions