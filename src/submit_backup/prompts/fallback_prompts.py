"""
Дефолтные промпты для RAG системы (фолбэк)
"""
from .topic_prompts import get_topic_prompt


def get_fallback_prompt(question: str, memory_text: str) -> str:
    """
    Получить дефолтный промпт для случаев, когда тема не определена
    
    Args:
        question: Вопрос пользователя
        memory_text: Текст памяти с сообщениями
        
    Returns:
        Дефолтный промпт для модели
    """
    return f"""ТЫ - ЭКСПЕРТ ПО АНАЛИЗУ ЛИЧНОЙ ИНФОРМАЦИИ! Найди точные факты о пользователе из его сообщений.

ЭТАП 1 - АНАЛИЗ ВОПРОСА:
Разбери вопрос "{question}" по словам. Пойми суть: что именно нужно найти?

ЭТАП 2 - СТРУКТУРА СЕССИЙ:
Сообщения идут хронологически: Сессия 1 → Сессия 25
При противоречиях - бери из ПОЗДНЕЙ сессии

ЭТАП 3 - ПРОВЕРКА:
Найди сообщения с ответом на вопрос.
Перечитай их 2-3 раза, дочитывай до конца фраз.

ЭТАП 4 - ОТВЕТ:
Найди ответ на "{question}". Отвечай кратко 1 предложением!

ПРАВИЛА:
• Найден факт → четкий ответ
• Косвенные упоминания тоже считаются
• Нет данных → "У меня нет такой информации"
• Несколько вариантов → через запятую

ФОРМАТ: Краткий ответ или "У меня нет такой информации"

СООБЩЕНИЯ:
{memory_text}"""


def get_prompt_by_topic(topic: str, question: str, memory_text: str) -> str:
    """
    Получить промпт для конкретной темы или дефолтный
    
    Args:
        topic: Название темы (может быть None для фолбэка)
        question: Вопрос пользователя
        memory_text: Текст памяти с сообщениями
        
    Returns:
        Подходящий промпт для модели
    """
    if topic and topic in get_all_topic_names():
        return get_topic_prompt(topic, question, memory_text)
    else:
        return get_fallback_prompt(question, memory_text)


def get_all_topic_names() -> list:
    """Получить список всех доступных тем"""
    from .topic_prompts import get_all_topic_names
    return get_all_topic_names()

