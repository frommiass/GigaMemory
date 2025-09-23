"""
Паттерны для извлечения фактов из текста
"""
import re
from typing import Dict, List, Tuple, Optional, Pattern
from .fact_models import FactType, FactRelation


# Базовые паттерны для извлечения фактов
FACT_PATTERNS: Dict[FactType, List[Pattern]] = {
    FactType.PERSONAL_NAME: [
        re.compile(r'(?:меня зовут|я\s*[-–—]\s*|мое имя\s*[-–—]?\s*)([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:зовут|называют)\s+(?:меня\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:я|мое имя)\s+([А-ЯЁ][а-яё]+)(?:\s|,|\.)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_AGE: [
        re.compile(r'(?:мне|мой возраст)\s*[-–—]?\s*(\d+)\s*(?:лет|год)', re.IGNORECASE),
        re.compile(r'(\d+)\s*(?:лет|год)(?:а|ов)?\s*мне', re.IGNORECASE),
        re.compile(r'я\s+(\d+)-?летн', re.IGNORECASE),
        re.compile(r'возраст\s*[-–—:]\s*(\d+)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_LOCATION: [
        re.compile(r'(?:живу|проживаю|нахожусь)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:из|родом из)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:мой город|место жительства)\s*[-–—:]\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.WORK_OCCUPATION: [
        re.compile(r'(?:работаю|тружусь|я)\s+([а-яё]+(?:ом|ером|ором|истом|ником|щиком|телем))', re.IGNORECASE),
        re.compile(r'(?:моя профессия|профессия|должность)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'я\s+(?:по профессии\s+)?([а-яё]+(?:ер|ор|ист|ник|щик|тель))', re.IGNORECASE),
    ],
    
    FactType.WORK_COMPANY: [
        re.compile(r'работаю\s+(?:в|на)\s+(?:компании\s+)?[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:компания|фирма|организация)\s*[-–—:]\s*[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:в|на)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)\s+работаю', re.IGNORECASE),
    ],
    
    FactType.FAMILY_SPOUSE: [
        re.compile(r'(?:жену?|мужа?)\s+зовут\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:женат|замужем)\s+(?:на|за)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:моя жена|мой муж)\s*[-–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:супруг|супруга)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.FAMILY_CHILDREN: [
        re.compile(r'(?:сын|дочь|дочка|ребенок)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:у меня\s+)?(\d+)\s+(?:детей|ребенок|ребенка|дочь|дочери|сын)', re.IGNORECASE),
        re.compile(r'(?:детей|детки)\s*[-–—:]\s*(\d+)', re.IGNORECASE),
    ],
    
    FactType.PET_NAME: [
        re.compile(r'(?:кот|кошк|пес|пёс|собак|питомец)\s+(?:по имени\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'зовут\s+(?:моего|мою)?\s*(?:кота|кошку|пса|собаку)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'([А-ЯЁ][а-яё]+)\s*[-–—]\s*(?:мой|моя)\s+(?:кот|кошка|пес|собака)', re.IGNORECASE),
    ],
    
    FactType.PET_TYPE: [
        re.compile(r'(?:у меня\s+(?:есть\s+)?|завел|держу)\s+(кот|кошка|пес|пёс|собака|хомяк|попугай|рыбки)', re.IGNORECASE),
        re.compile(r'(?:мой|моя|мои)\s+(кот|кошка|пес|собака|хомяк|попугай|рыбки)', re.IGNORECASE),
    ],
    
    FactType.PET_BREED: [
        re.compile(r'(?:порода|породы)\s*[-–—:]\s*([а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'([а-яё]+(?:\s+[а-яё]+)*)\s+(?:порода|породы)', re.IGNORECASE),
        re.compile(r'(?:кот|кошка|пес|собака)\s+породы\s+([а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
    ],
    
    FactType.HOBBY_SPORT: [
        re.compile(r'(?:играю|занимаюсь)\s+(?:в\s+)?([а-яё]+(?:ом|болом|сом|кетболом))', re.IGNORECASE),
        re.compile(r'(?:люблю|обожаю|увлекаюсь)\s+([а-яё]+(?:ом|бол|кетбол|инг|анием))', re.IGNORECASE),
        re.compile(r'хожу\s+(?:на|в)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.HOBBY_ACTIVITY: [
        re.compile(r'(?:увлекаюсь|занимаюсь|люблю)\s+([а-яё]+(?:ием|анием|кой|ством))', re.IGNORECASE),
        re.compile(r'(?:мое хобби|хобби)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.PREFERENCE_FOOD: [
        re.compile(r'(?:люблю|обожаю|предпочитаю)\s+(?:есть\s+)?([а-яё]+(?:у|ы|и))', re.IGNORECASE),
        re.compile(r'(?:любимая еда|блюдо)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:не люблю|не ем|терпеть не могу)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.EDUCATION_INSTITUTION: [
        re.compile(r'(?:учился|училась|окончил|закончил)\s+([А-ЯЁ][а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:университет|институт|вуз|колледж)\s*[-–—:]\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:в|во)\s+([А-ЯЁ][А-ЯЁа-яё]+(?:\s+[а-яё]+)*)\s+(?:учился|училась)', re.IGNORECASE),
    ],
    
    FactType.HEALTH_CONDITION: [
        re.compile(r'(?:болею|страдаю|есть|имею)\s+([а-яё]+(?:ией|ией|зом|тис|ия))', re.IGNORECASE),
        re.compile(r'(?:диагноз|заболевание)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:проблемы с|болит|болят)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.EVENT_TRAVEL: [
        re.compile(r'(?:был|была|ездил|летал|путешествовал)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:поеду|полечу|собираюсь)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:отпуск|поездка)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
}


# Паттерны для извлечения отношений
RELATION_PATTERNS: Dict[FactRelation, List[Pattern]] = {
    FactRelation.IS: [
        re.compile(r'(?:я|это|являюсь|есть)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.HAS: [
        re.compile(r'(?:у меня|имею|есть)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.WORKS_AS: [
        re.compile(r'работаю\s+(?:как\s+)?(.+)', re.IGNORECASE),
    ],
    FactRelation.WORKS_AT: [
        re.compile(r'работаю\s+(?:в|на)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.LIVES_IN: [
        re.compile(r'живу\s+(?:в|на)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.LIKES: [
        re.compile(r'(?:люблю|нравится|обожаю)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.OWNS: [
        re.compile(r'(?:владею|имею|есть)\s+(.+)', re.IGNORECASE),
    ],
}


# Временные маркеры
TEMPORAL_PATTERNS = {
    'past': [
        re.compile(r'(?:раньше|прежде|когда-то|в прошлом|ранее)', re.IGNORECASE),
        re.compile(r'(?:был|была|были)\s+(.+)', re.IGNORECASE),
        re.compile(r'(?:в|во?)\s+(\d{4})\s*(?:году)?', re.IGNORECASE),
    ],
    'future': [
        re.compile(r'(?:буду|будет|планирую|собираюсь|хочу)', re.IGNORECASE),
        re.compile(r'(?:в следующем|через|скоро|вскоре)', re.IGNORECASE),
    ],
    'current': [
        re.compile(r'(?:сейчас|теперь|в данный момент|на данный момент|сегодня)', re.IGNORECASE),
    ],
}


def compile_patterns() -> Dict[str, List[Pattern]]:
    """Компилирует все паттерны для оптимизации"""
    compiled = {}
    for fact_type, patterns in FACT_PATTERNS.items():
        compiled[fact_type.value] = patterns
    return compiled


def get_fact_pattern(fact_type: FactType) -> List[Pattern]:
    """Возвращает паттерны для конкретного типа факта"""
    return FACT_PATTERNS.get(fact_type, [])


def extract_with_pattern(text: str, pattern: Pattern) -> Optional[str]:
    """
    Извлекает данные по паттерну
    
    Args:
        text: Текст для поиска
        pattern: Регулярное выражение
        
    Returns:
        Извлеченное значение или None
    """
    match = pattern.search(text)
    if match and len(match.groups()) > 0:
        return match.group(1).strip()
    return None


def extract_all_with_patterns(text: str, patterns: List[Pattern]) -> List[str]:
    """
    Извлекает все совпадения по списку паттернов
    
    Args:
        text: Текст для поиска
        patterns: Список регулярных выражений
        
    Returns:
        Список извлеченных значений
    """
    results = []
    for pattern in patterns:
        value = extract_with_pattern(text, pattern)
        if value and value not in results:
            results.append(value)
    return results


def detect_temporal_context(text: str) -> Optional[str]:
    """
    Определяет временной контекст высказывания
    
    Args:
        text: Текст для анализа
        
    Returns:
        'past', 'future', 'current' или None
    """
    text_lower = text.lower()
    
    for temporal_type, patterns in TEMPORAL_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text_lower):
                return temporal_type
    
    return 'current'  # По умолчанию считаем текущим


def normalize_value(value: str, fact_type: FactType) -> str:
    """
    Нормализует извлеченное значение
    
    Args:
        value: Исходное значение
        fact_type: Тип факта
        
    Returns:
        Нормализованное значение
    """
    value = value.strip()
    
    # Убираем лишние пробелы
    value = re.sub(r'\s+', ' ', value)
    
    # Специфичная нормализация по типам
    if fact_type in [FactType.PERSONAL_NAME, FactType.FAMILY_SPOUSE, 
                     FactType.PET_NAME, FactType.PERSONAL_LOCATION]:
        # Капитализация имен и названий
        value = value.title()
    
    elif fact_type == FactType.PERSONAL_AGE:
        # Оставляем только число
        match = re.search(r'\d+', value)
        if match:
            value = match.group()
    
    elif fact_type in [FactType.PET_TYPE, FactType.HOBBY_SPORT]:
        # Приводим к нижнему регистру
        value = value.lower()
    
    # Убираем знаки препинания в конце
    value = value.rstrip('.,!?;:')
    
    return value


def confidence_from_pattern_match(pattern_index: int, total_patterns: int) -> float:
    """
    Рассчитывает уверенность на основе индекса паттерна
    
    Args:
        pattern_index: Индекс сработавшего паттерна (0 = самый точный)
        total_patterns: Общее количество паттернов
        
    Returns:
        Уверенность от 0.5 до 1.0
    """
    if total_patterns <= 1:
        return 0.9
    
    # Линейное убывание уверенности
    base_confidence = 1.0
    decrease_step = 0.3 / total_patterns
    confidence = base_confidence - (pattern_index * decrease_step)
    
    return max(0.5, min(1.0, confidence))



