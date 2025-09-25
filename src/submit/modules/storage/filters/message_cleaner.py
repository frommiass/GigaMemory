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
    
    DEPRECATED: Используйте MessageFilter из src.submit.core.message_filter
    
    Args:
        messages: Список сообщений для очистки
        
    Returns:
        Список очищенных сообщений пользователя
    """
    # Импортируем центральный фильтр
    from ..message_filter import message_filter
    return message_filter.filter_messages(messages)


def clean_user_messages(messages: List[Message]) -> List[Message]:
    """
    Алиас для обратной совместимости с существующим кодом
    
    DEPRECATED: Используйте MessageFilter из src.submit.core.message_filter
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
    
    # Сначала проверяем явные маркеры копипаста
    explicit_copypaste = [
        'скопировал', 'вставил', 'файл', 'документ', 'текст из', 'ссылка',
        'прикрепил', 'отправил', 'загрузил', 'приложил', 'добавил файл',
        'переведи', 'перескажи', 'проанализируй', 'исправь', 'проверь',
        'объясни', 'разбери', 'сделай', 'помоги с', 'реши', 'найди ошибки',
        'оптимизируй', 'улучши', 'переделай', 'доработай', 'доделай',
        'расскажи о', 'опиши', 'что это', 'о чём', 'суть', 'смысл',
        'содержание', 'краткий пересказ', 'в двух словах', 'коротко',
        'можешь рассказать', 'что там написано', 'про что',
        'найди информацию', 'поищи', 'посмотри', 'проверь данные',
        'сравни', 'сопоставь', 'выдели главное', 'основные моменты',
        'ключевые точки', 'важные детали', 'нашёл', 'нашла',
        'тут статья', 'вот ссылка', 'посмотри на', 'это интересно',
        'прочитал', 'прочитала', 'увидел', 'увидела', 'нашёл', 'нашла',
        'попалось', 'наткнулся', 'наткнулась', 'встретил', 'встретила',
        # Новые маркеры
        'напиши', 'расскажи', 'рилз', 'reels', 'сторис', 'stories',
        'объясни', 'разъясни', 'растолкуй', 'покажи', 'продемонстрируй',
        'сделай', 'создай', 'нарисуй', 'сочини', 'придумай',
        'помоги', 'подскажи', 'совет', 'рекомендация',
        'переведи', 'перескажи', 'кратко', 'коротко', 'суть', 'главное',
        'опиши', 'описание', 'характеристика', 'свойства', 'особенности',
        'найди', 'исправь', 'проверь'
    ]
    
    # Проверяем явные маркеры
    if _check_markers(content_lower, frozenset(explicit_copypaste)):
        return True
    
    # Проверяем только односложные маркеры копипаста
    # (фразы из двух слов убраны)
    
    # Проверяем длину сообщения - длинные сообщения часто копипаст
    if len(content) > 300:
        # Дополнительные проверки для длинных сообщений
        # Проверяем наличие технических терминов
        tech_terms = [
            'автомобиль', 'автомобилю', 'мотор', 'двигатель', 'кузов', 'комплектация',
            'технические', 'характеристики', 'параметры', 'спецификация',
            'производство', 'выпуск', 'модель', 'поколение', 'платформа',
            'дизайн', 'внешний вид', 'интерьер', 'салон', 'багажник',
            'безопасность', 'подушки', 'система', 'электроника', 'автомат',
            'механика', 'коробка', 'передач', 'привод', 'подвеска',
            'тормоза', 'рулевое', 'управление', 'кондиционер', 'климат',
            'мультимедиа', 'навигация', 'bluetooth', 'usb', 'aux',
            'стереосистема', 'динамики', 'магнитола', 'радио', 'cd',
            'renault', 'hyundai', 'kia', 'logan', 'solaris', 'rio'
        ]
        
        if _check_markers(content_lower, frozenset(tech_terms)):
            return True
        
        # Проверяем структуру текста - много заголовков, списков
        structure_indicators = [
            'первое поколение', 'второе поколение', 'третье поколение',
            'история', 'развитие', 'эволюция', 'технические характеристики',
            'преимущества', 'недостатки', 'особенности', 'отличия',
            'сравнение', 'анализ', 'обзор', 'тест', 'испытание'
        ]
        
        if _check_markers(content_lower, frozenset(structure_indicators)):
            return True
    
    # Дополнительная проверка для очень длинных сообщений с техническими терминами
    if len(content) > 500:
        tech_word_count = 0
        for term in tech_terms:
            if term in content_lower:
                tech_word_count += 1
        
        # Если найдено много технических терминов - это копипаст
        if tech_word_count >= 5:
            return True
    
    # Проверка для очень длинных сообщений (>1000 символов) с множественными техническими терминами
    if len(content) > 1000:
        tech_word_count = 0
        for term in tech_terms:
            if term in content_lower:
                tech_word_count += 1
        
        # Если найдено много технических терминов в очень длинном сообщении - это точно копипаст
        if tech_word_count >= 3:
            return True
    
    # Проверяем технические маркеры только в контексте копипаста
    tech_copypaste = [
        'код', 'программа', 'скрипт', 'функция', 'алгоритм', 'формула',
        'таблица', 'график', 'диаграмма', 'схема', 'чертеж',
        'задача', 'упражнение', 'домашка', 'контрольная', 'экзамен',
        'курсовая', 'диплом', 'реферат', 'эссе', 'сочинение',
        'резюме', 'письмо', 'договор', 'отчет', 'презентация',
        'инструкция', 'руководство', 'мануал', 'техзадание',
        'статья', 'новость', 'пост', 'комментарий', 'отзыв',
        'описание товара', 'вакансия', 'объявление', 'блог'
    ]
    
    # Проверяем технические маркеры только если есть контекст копипаста
    if _check_markers(content_lower, frozenset(tech_copypaste)):
        # Дополнительная проверка - есть ли контекст копипаста
        copypaste_context = [
            'помоги', 'сделай', 'переведи', 'исправь', 'найди',
            'объясни', 'разбери', 'проанализируй', 'проверь'
        ]
        if _check_markers(content_lower, frozenset(copypaste_context)):
            return True
    
    # Проверяем развлекательный контент только в контексте копипаста
    entertainment_copypaste = [
        'видео', 'ролик', 'фильм', 'сериал', 'книга', 'роман'
    ]
    
    if _check_markers(content_lower, frozenset(entertainment_copypaste)):
        # Дополнительная проверка - есть ли контекст копипаста
        copypaste_context = [
            'посмотри', 'расскажи', 'перескажи', 'что там', 'про что',
            'интересно', 'понравилось', 'рекомендую'
        ]
        if _check_markers(content_lower, frozenset(copypaste_context)):
            return True
    
    return False


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
