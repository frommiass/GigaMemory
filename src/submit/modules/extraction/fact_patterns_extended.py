# modules/extraction/fact_patterns_extended.py
"""
Расширенные паттерны для извлечения фактов с поддержкой info_updating
"""

import re
from typing import Dict, List, Pattern, Optional, Tuple
from .fact_models import FactType, FactRelation


class InfoUpdatePatterns:
    """
    Специальные паттерны для обнаружения обновлений информации
    """
    
    # Паттерны для определения временных изменений
    TEMPORAL_CHANGE_PATTERNS = {
        'past_to_present': [
            re.compile(r'(?:был|была)\s+([а-яё]+),?\s+(?:теперь|сейчас|стал|стала)\s+([а-яё]+)', re.IGNORECASE),
            re.compile(r'(?:раньше|прежде)\s+([а-яё]+),?\s+(?:теперь|сейчас)\s+([а-яё]+)', re.IGNORECASE),
            re.compile(r'(?:изменилось|поменялось):\s*([а-яё\s]+)\s+на\s+([а-яё\s]+)', re.IGNORECASE),
        ],
        
        'status_change': [
            re.compile(r'(?:больше не|уже не)\s+([а-яё]+)', re.IGNORECASE),
            re.compile(r'(?:перестал|перестала)\s+(?:быть\s+)?([а-яё]+)', re.IGNORECASE),
            re.compile(r'(?:стал|стала|становлюсь)\s+([а-яё]+)', re.IGNORECASE),
        ],
        
        'life_events': [
            # Семейные события
            re.compile(r'(?:женился|женюсь)\s+(?:на\s+)?([А-ЯЁ][а-яё]+)?', re.IGNORECASE),
            re.compile(r'(?:вышла замуж|выхожу замуж)\s+(?:за\s+)?([А-ЯЁ][а-яё]+)?', re.IGNORECASE),
            re.compile(r'(?:развелся|развелась|разводимся|расстались)', re.IGNORECASE),
            re.compile(r'(?:овдовел|овдовела)', re.IGNORECASE),
            
            # Рождение детей
            re.compile(r'(?:родился|родилась)\s+(?:сын|дочь|ребенок)\s*(?:по имени\s+)?([А-ЯЁ][а-яё]+)?', re.IGNORECASE),
            re.compile(r'(?:стал|стала)\s+(?:отцом|мамой|родителем)', re.IGNORECASE),
            
            # Переезд
            re.compile(r'(?:переехал|переехала|переезжаю)\s+(?:в|из)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
            re.compile(r'(?:уехал|уехала)\s+(?:в|из)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
            
            # Работа
            re.compile(r'(?:уволился|уволилась|ушел|ушла)\s+(?:с работы|из)\s*([А-ЯЁ][а-яё]+)?', re.IGNORECASE),
            re.compile(r'(?:устроился|устроилась)\s+(?:на работу\s+)?(?:в\s+)?([А-ЯЁ][а-яё]+)?', re.IGNORECASE),
            re.compile(r'(?:повысили|понизили)\s+(?:до\s+)?([а-яё]+)?', re.IGNORECASE),
            re.compile(r'(?:сменил|сменила)\s+(?:работу|профессию)\s+(?:на\s+)?([а-яё]+)?', re.IGNORECASE),
        ]
    }
    
    # Маркеры актуальности информации
    ACTUALITY_MARKERS = {
        'current': [
            'сейчас', 'теперь', 'в данный момент', 'на данный момент',
            'в настоящее время', 'сегодня', 'на сегодняшний день',
            'актуально', 'по состоянию на'
        ],
        'past': [
            'был', 'была', 'были', 'раньше', 'прежде', 'когда-то',
            'в прошлом', 'ранее', 'до этого', 'до того как'
        ],
        'future': [
            'буду', 'будет', 'планирую', 'собираюсь', 'хочу',
            'намерен', 'в планах', 'скоро', 'вскоре', 'завтра'
        ]
    }
    
    @classmethod
    def detect_info_update_type(cls, text: str) -> Tuple[str, Dict]:
        """
        Определяет тип обновления информации
        
        Returns:
            Tuple[тип_обновления, извлеченные_данные]
        """
        text_lower = text.lower()
        
        # Проверяем изменения статуса
        for pattern in cls.TEMPORAL_CHANGE_PATTERNS['status_change']:
            match = pattern.search(text)
            if match:
                return 'status_change', {'new_status': match.group(1) if match.groups() else None}
        
        # Проверяем жизненные события
        for pattern in cls.TEMPORAL_CHANGE_PATTERNS['life_events']:
            match = pattern.search(text)
            if match:
                event_type = cls._determine_event_type(text)
                return 'life_event', {
                    'event': event_type,
                    'details': match.group(1) if match.groups() else None
                }
        
        # Проверяем изменения "было -> стало"
        for pattern in cls.TEMPORAL_CHANGE_PATTERNS['past_to_present']:
            match = pattern.search(text)
            if match:
                return 'value_change', {
                    'old_value': match.group(1),
                    'new_value': match.group(2)
                }
        
        return 'none', {}
    
    @classmethod
    def _determine_event_type(cls, text: str) -> str:
        """Определяет тип жизненного события"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['женился', 'женюсь', 'вышла замуж', 'свадьба']):
            return 'marriage'
        elif any(word in text_lower for word in ['развелся', 'развелась', 'развод']):
            return 'divorce'
        elif any(word in text_lower for word in ['родился', 'родилась', 'ребенок']):
            return 'child_birth'
        elif any(word in text_lower for word in ['переехал', 'переехала', 'переезд']):
            return 'relocation'
        elif any(word in text_lower for word in ['уволился', 'уволилась', 'устроился', 'устроилась']):
            return 'job_change'
        elif any(word in text_lower for word in ['умер', 'умерла', 'скончался']):
            return 'death'
        
        return 'other'
    
    @classmethod
    def extract_updated_facts(cls, text: str, fact_type: FactType) -> List[Dict]:
        """
        Извлекает обновленные факты определенного типа
        """
        updated_facts = []
        update_type, update_data = cls.detect_info_update_type(text)
        
        if update_type == 'none':
            return []
        
        # Маппинг событий на типы фактов
        event_to_fact_type = {
            'marriage': FactType.FAMILY_STATUS,
            'divorce': FactType.FAMILY_STATUS,
            'child_birth': FactType.FAMILY_CHILDREN,
            'relocation': FactType.PERSONAL_LOCATION,
            'job_change': FactType.WORK_COMPANY,
        }
        
        if update_type == 'life_event':
            event = update_data.get('event')
            if event in event_to_fact_type:
                target_fact_type = event_to_fact_type[event]
                
                # Определяем новое значение
                new_value = cls._determine_new_value(event, update_data)
                
                if new_value and target_fact_type == fact_type:
                    updated_facts.append({
                        'type': target_fact_type,
                        'value': new_value,
                        'is_current': True,
                        'confidence': 0.95,
                        'update_type': event
                    })
        
        elif update_type == 'value_change':
            # Для изменений "было -> стало"
            old_value = update_data.get('old_value')
            new_value = update_data.get('new_value')
            
            if new_value:
                updated_facts.append({
                    'type': fact_type,
                    'value': new_value,
                    'old_value': old_value,
                    'is_current': True,
                    'confidence': 0.9,
                    'update_type': 'value_change'
                })
        
        return updated_facts
    
    @classmethod
    def _determine_new_value(cls, event: str, data: Dict) -> Optional[str]:
        """Определяет новое значение для события"""
        if event == 'marriage':
            return 'женат' if data.get('details') else 'в браке'
        elif event == 'divorce':
            return 'разведен'
        elif event == 'child_birth':
            return data.get('details', 'есть ребенок')
        elif event == 'relocation':
            return data.get('details')
        elif event == 'job_change':
            return data.get('details')
        
        return None


class EnhancedFactPatterns:
    """
    Улучшенные паттерны для извлечения сложных фактов
    """
    
    # Паттерны для составных фактов
    COMPOSITE_PATTERNS = {
        # Полное имя (имя + фамилия + отчество)
        'full_name': [
            re.compile(r'(?:я|меня зовут|мое полное имя)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?', re.IGNORECASE),
            re.compile(r'(?:ФИО|фио)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?', re.IGNORECASE),
        ],
        
        # Полный адрес
        'full_address': [
            re.compile(r'(?:живу|проживаю|адрес)\s*[:–—]?\s*(?:г\.?\s*)?([А-ЯЁ][а-яё]+),?\s*(?:ул\.?\s*)?([А-ЯЁ][а-яё]+),?\s*(?:д\.?\s*)?(\d+)', re.IGNORECASE),
            re.compile(r'([А-ЯЁ][а-яё]+)\s+(?:город|г\.),?\s*([а-яё]+)\s+(?:улица|ул\.),?\s*(?:дом\s+)?(\d+)', re.IGNORECASE),
        ],
        
        # Образование с годом
        'education_with_year': [
            re.compile(r'(?:окончил|закончил)\s+([А-ЯЁ][А-ЯЁа-яё]+(?:\s+[а-яё]+)*)\s+в\s+(\d{4})', re.IGNORECASE),
            re.compile(r'(?:выпускник|выпускница)\s+([А-ЯЁ][А-ЯЁа-яё]+)\s+(\d{4})\s+года', re.IGNORECASE),
        ],
        
        # Опыт работы
        'work_experience': [
            re.compile(r'(?:работаю|стаж)\s+([а-яё]+)\s+(\d+)\s+(?:лет|год)', re.IGNORECASE),
            re.compile(r'(\d+)\s+(?:лет|год)\s+(?:работаю|опыта)\s+([а-яё]+)', re.IGNORECASE),
        ],
        
        # Зарплата с валютой
        'salary_with_currency': [
            re.compile(r'(?:зарплата|получаю|зарабатываю)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)\s*(руб|долл|евро|₽|\$|€)', re.IGNORECASE),
            re.compile(r'(\d+(?:\s?\d{3})*)\s*(руб|долл|евро|₽|\$|€)\s+(?:в месяц|в год)', re.IGNORECASE),
        ]
    }
    
    # Контекстные паттерны (требуют анализа контекста)
    CONTEXTUAL_PATTERNS = {
        'family_relations': {
            'patterns': [
                re.compile(r'(?:мой|моя|мои)\s+([а-яё]+)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                re.compile(r'([А-ЯЁ][а-яё]+)\s*[-–—]\s*(?:мой|моя)\s+([а-яё]+)', re.IGNORECASE),
            ],
            'context_words': {
                'spouse': ['жена', 'муж', 'супруг', 'супруга'],
                'children': ['сын', 'дочь', 'дочка', 'ребенок', 'дети'],
                'parents': ['мама', 'папа', 'отец', 'мать', 'родитель'],
                'siblings': ['брат', 'сестра'],
            }
        },
        
        'professional_skills': {
            'patterns': [
                re.compile(r'(?:владею|знаю|умею)\s+([A-Za-zа-яё\+\#\s,]+)', re.IGNORECASE),
                re.compile(r'(?:навыки|skills)\s*[:–—]?\s*([A-Za-zа-яё\+\#\s,]+)', re.IGNORECASE),
            ],
            'context_words': ['программирование', 'язык', 'технология', 'framework']
        }
    }
    
    @classmethod
    def extract_composite_facts(cls, text: str) -> List[Dict]:
        """
        Извлекает составные факты
        """
        composite_facts = []
        
        # Извлечение полного имени
        for pattern in cls.COMPOSITE_PATTERNS['full_name']:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                composite_facts.append({
                    'type': 'full_name',
                    'first_name': groups[0],
                    'last_name': groups[1],
                    'middle_name': groups[2] if len(groups) > 2 else None,
                    'confidence': 0.95
                })
                break
        
        # Извлечение полного адреса
        for pattern in cls.COMPOSITE_PATTERNS['full_address']:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                composite_facts.append({
                    'type': 'full_address',
                    'city': groups[0],
                    'street': groups[1],
                    'house': groups[2],
                    'confidence': 0.85
                })
                break
        
        # Извлечение образования с годом
        for pattern in cls.COMPOSITE_PATTERNS['education_with_year']:
            match = pattern.search(text)
            if match:
                composite_facts.append({
                    'type': 'education',
                    'institution': match.group(1),
                    'year': match.group(2),
                    'confidence': 0.9
                })
        
        # Извлечение опыта работы
        for pattern in cls.COMPOSITE_PATTERNS['work_experience']:
            match = pattern.search(text)
            if match:
                # Определяем порядок групп
                if match.group(1).isdigit():
                    years, profession = match.group(1), match.group(2)
                else:
                    profession, years = match.group(1), match.group(2)
                
                composite_facts.append({
                    'type': 'work_experience',
                    'profession': profession,
                    'years': years,
                    'confidence': 0.85
                })
        
        return composite_facts
    
    @classmethod
    def extract_contextual_facts(cls, text: str) -> List[Dict]:
        """
        Извлекает факты, требующие анализа контекста
        """
        contextual_facts = []
        
        # Анализ семейных отношений
        family_context = cls.CONTEXTUAL_PATTERNS['family_relations']
        for pattern in family_context['patterns']:
            matches = pattern.finditer(text)
            for match in matches:
                relation_word = match.group(1).lower() if match.group(1) else match.group(2).lower()
                name = match.group(2) if match.group(1) else match.group(1)
                
                # Определяем тип отношения
                relation_type = None
                for rel_type, words in family_context['context_words'].items():
                    if any(word in relation_word for word in words):
                        relation_type = rel_type
                        break
                
                if relation_type and name:
                    contextual_facts.append({
                        'type': f'family_{relation_type}',
                        'name': name,
                        'relation': relation_word,
                        'confidence': 0.8
                    })
        
        # Анализ профессиональных навыков
        skills_context = cls.CONTEXTUAL_PATTERNS['professional_skills']
        for pattern in skills_context['patterns']:
            match = pattern.search(text)
            if match:
                skills_text = match.group(1)
                # Разделяем навыки по запятым
                skills = [s.strip() for s in re.split(r'[,;]', skills_text)]
                
                for skill in skills:
                    if skill and len(skill) > 1:
                        contextual_facts.append({
                            'type': 'professional_skill',
                            'skill': skill,
                            'confidence': 0.75
                        })
        
        return contextual_facts


def create_enhanced_patterns() -> Dict:
    """
    Создает расширенный набор паттернов для извлечения
    """
    from .fact_patterns import FACT_PATTERNS
    
    # Копируем базовые паттерны
    enhanced = FACT_PATTERNS.copy()
    
    # Добавляем новые паттерны для info_updating
    enhanced[FactType.PERSONAL_NAME].extend([
        re.compile(r'(?:теперь меня зовут|изменил имя на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:стал|стала)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ])
    
    enhanced[FactType.PERSONAL_AGE].extend([
        re.compile(r'(?:мне уже|стукнуло|исполнилось)\s+(\d+)', re.IGNORECASE),
        re.compile(r'(?:справил|отметил)\s+(\d+)[-\s]?летие', re.IGNORECASE),
    ])
    
    enhanced[FactType.PERSONAL_LOCATION].extend([
        re.compile(r'(?:переехал|переехала|переезжаю)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:теперь живу|сейчас живу)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:вернулся|вернулась)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ])
    
    enhanced[FactType.WORK_COMPANY].extend([
        re.compile(r'(?:перешел|перешла)\s+(?:работать\s+)?(?:в|к)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:теперь работаю|сейчас работаю)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:устроился|устроилась)\s+(?:в|к)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ])
    
    enhanced[FactType.FAMILY_STATUS].extend([
        re.compile(r'(?:женился|женюсь|женился недавно|женился наконец)', re.IGNORECASE),
        re.compile(r'(?:вышла замуж|выхожу замуж|наконец замужем)', re.IGNORECASE),
        re.compile(r'(?:развелся|развелась|разводимся|в разводе)', re.IGNORECASE),
        re.compile(r'(?:овдовел|овдовела|вдовец|вдова)', re.IGNORECASE),
        re.compile(r'(?:теперь|сейчас|стал|стала)\s+(холост|не женат|не замужем|одинок)', re.IGNORECASE),
    ])
    
    # Новый тип для детей с учетом обновлений
    enhanced[FactType.FAMILY_CHILDREN].extend([
        re.compile(r'(?:родился|родилась)\s+(?:сын|дочь|дочка|ребенок)', re.IGNORECASE),
        re.compile(r'(?:стал|стала)\s+(?:папой|мамой|отцом|матерью)', re.IGNORECASE),
        re.compile(r'(?:теперь у меня|появился|появилась)\s+(?:сын|дочь|ребенок)', re.IGNORECASE),
        re.compile(r'(?:жду|ждем)\s+(?:ребенка|малыша|пополнения)', re.IGNORECASE),
    ])
    
    # Паттерны для питомцев с учетом изменений
    enhanced[FactType.PET_TYPE].extend([
        re.compile(r'(?:завел|завела|взял|взяла|появился)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:теперь у меня|купил|купила)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:подарили|привез|привезла)\s+([а-яё]+)', re.IGNORECASE),
    ])
    
    return enhanced


def detect_question_type(text: str) -> str:
    """
    Определяет тип вопроса для оптимизации извлечения
    """
    text_lower = text.lower()
    
    # Вопросы об обновлении информации
    if any(marker in text_lower for marker in [
        'изменилось', 'обновить', 'теперь', 'сейчас',
        'новый', 'новая', 'актуальн', 'последн'
    ]):
        return 'info_updating'
    
    # Вопросы о личной информации
    if any(marker in text_lower for marker in [
        'как тебя зовут', 'твое имя', 'сколько лет',
        'где живешь', 'откуда ты'
    ]):
        return 'personal_info'
    
    # Вопросы о работе
    if any(marker in text_lower for marker in [
        'где работаешь', 'кем работаешь', 'профессия',
        'должность', 'компания'
    ]):
        return 'work_info'
    
    # Вопросы о семье
    if any(marker in text_lower for marker in [
        'женат', 'замужем', 'дети', 'семья',
        'жена', 'муж', 'супруг'
    ]):
        return 'family_info'
    
    return 'general'