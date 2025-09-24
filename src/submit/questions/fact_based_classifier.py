from typing import Optional, Tuple, List, Set
from ..extraction.fact_models import FactType
import re

class FactBasedQuestionClassifier:
    """Классификатор вопросов на основе типов фактов из extraction"""
    
    def __init__(self):
        # Маппинг ключевых слов вопроса -> FactType
        self.question_patterns = {
            # ЛИЧНАЯ ИНФОРМАЦИЯ
            FactType.PERSONAL_NAME: [
                r'как\s+(?:меня\s+)?зовут',
                r'(?:мое|моё)\s+имя',
                r'кто\s+я(?:\s+такой)?',
            ],
            
            FactType.PERSONAL_AGE: [
                r'сколько\s+(?:мне\s+)?лет',
                r'(?:мой\s+)?возраст',
                r'когда\s+(?:я\s+)?родил',
            ],
            
            FactType.PERSONAL_LOCATION: [
                r'где\s+(?:я\s+)?живу',
                r'(?:мой\s+)?(?:город|адрес)',
                r'откуда\s+я',
            ],
            
            # РАБОТА
            FactType.WORK_OCCUPATION: [
                r'кем\s+(?:я\s+)?работа',
                r'(?:моя\s+)?(?:профессия|специальность)',
                r'чем\s+(?:я\s+)?занима',
            ],
            
            FactType.WORK_COMPANY: [
                r'где\s+(?:я\s+)?работа',
                r'(?:моя\s+)?(?:компания|фирма|организация)',
                r'в\s+какой\s+(?:компании|организации)',
            ],
            
            FactType.WORK_POSITION: [
                r'(?:моя\s+)?должность',
                r'кем\s+(?:я\s+)?(?:являюсь|работаю)\s+(?:в|на)',
                r'(?:мой\s+)?пост',
            ],
            
            FactType.WORK_SALARY: [
                r'(?:моя\s+)?(?:зарплата|заработок|доход)',
                r'сколько\s+(?:я\s+)?(?:зарабатыва|получа)',
            ],
            
            # СЕМЬЯ
            FactType.FAMILY_STATUS: [
                r'(?:я\s+)?(?:женат|замужем)',
                r'(?:мое|моё)\s+семейное\s+положение',
                r'есть\s+ли\s+(?:у\s+меня\s+)?(?:жена|муж|супруг)',
            ],
            
            FactType.FAMILY_SPOUSE: [
                r'как\s+зовут\s+(?:мою\s+)?(?:жену|мужа|супруг)',
                r'кто\s+(?:моя\s+жена|мой\s+муж)',
                r'(?:имя|фио)\s+(?:жены|мужа|супруг)',
            ],
            
            FactType.FAMILY_CHILDREN: [
                r'(?:сколько\s+)?(?:у\s+меня\s+)?дет(?:ей|и)',
                r'есть\s+ли\s+(?:у\s+меня\s+)?(?:сын|дочь|дети)',
                r'(?:имена|возраст)\s+(?:моих\s+)?детей',
            ],
            
            # ТРАНСПОРТ
            FactType.TRANSPORT_CAR_BRAND: [
                r'(?:какая\s+)?(?:у\s+меня\s+)?(?:машина|авто|тачка)',
                r'(?:марка|бренд)\s+(?:моей\s+)?(?:машины|авто)',
                r'на\s+чем\s+(?:я\s+)?(?:езжу|катаюсь)',
            ],
            
            FactType.TRANSPORT_CAR_MODEL: [
                r'(?:модель)\s+(?:моей\s+)?(?:машины|авто)',
                r'какая\s+(?:у\s+меня\s+)?модель',
            ],
            
            # ФИНАНСЫ
            FactType.FINANCE_INCOME: [
                r'(?:мой\s+)?(?:доход|заработок)',
                r'сколько\s+(?:я\s+)?(?:получаю|имею)',
            ],
            
            FactType.FINANCE_SAVINGS: [
                r'(?:мои\s+)?(?:сбережения|накопления)',
                r'сколько\s+(?:я\s+)?(?:накопил|отложил)',
            ],
            
            FactType.FINANCE_BANK: [
                r'(?:мой\s+)?банк',
                r'где\s+(?:я\s+)?(?:храню|держу)\s+деньги',
                r'(?:банковские\s+)?карты',
            ],
            
            # ЕДА И НАПИТКИ
            FactType.FOOD_FAVORITE: [
                r'(?:моя\s+)?любимая\s+(?:еда|пища)',
                r'что\s+(?:я\s+)?(?:люблю\s+)?(?:есть|кушать)',
            ],
            
            FactType.DRINK_COFFEE: [
                r'(?:какой\s+)?кофе\s+(?:я\s+)?(?:пью|люблю)',
                r'(?:пью\s+ли\s+я\s+)?кофе',
            ],
            
            # ПУТЕШЕСТВИЯ
            FactType.TRAVEL_COUNTRY: [
                r'(?:где|куда)\s+(?:я\s+)?(?:путешествовал|ездил|летал)',
                r'(?:какие\s+)?страны\s+(?:я\s+)?(?:посетил|был)',
            ],
            
            # СПОРТ
            FactType.SPORT_TYPE: [
                r'(?:каким\s+)?спортом\s+(?:я\s+)?(?:занимаюсь|увлекаюсь)',
                r'(?:мои\s+)?(?:тренировки|физические\s+нагрузки)',
            ],
            
            FactType.SPORT_GYM: [
                r'(?:мой\s+)?(?:спортзал|фитнес|зал)',
                r'где\s+(?:я\s+)?(?:тренируюсь|занимаюсь)',
            ],
            
            # ОБРАЗОВАНИЕ
            FactType.EDUCATION_INSTITUTION: [
                r'где\s+(?:я\s+)?(?:учился|училась|учусь)',
                r'(?:мой\s+)?(?:университет|институт|вуз)',
                r'(?:какое\s+)?(?:у\s+меня\s+)?образование',
            ],
            
            FactType.EDUCATION_SPECIALITY: [
                r'(?:моя\s+)?специальность',
                r'(?:кем|на\s+кого)\s+(?:я\s+)?(?:учился|училась)',
            ],
            
            # ХОББИ
            FactType.HOBBY_ACTIVITY: [
                r'(?:мои\s+)?(?:хобби|увлечения|интересы)',
                r'чем\s+(?:я\s+)?(?:увлекаюсь|интересуюсь)',
                r'(?:мое|моё)\s+(?:любимое\s+)?(?:занятие|дело)',
            ],
            
            # ТЕХНОЛОГИИ
            FactType.TECH_PHONE: [
                r'(?:мой\s+)?(?:телефон|смартфон)',
                r'(?:какой\s+)?(?:у\s+меня\s+)?(?:айфон|самсунг|xiaomi)',
            ],
            
            # ЗДОРОВЬЕ
            FactType.HEALTH_CONDITION: [
                r'(?:мое|моё)\s+(?:здоровье|самочувствие)',
                r'(?:чем\s+)?(?:я\s+)?(?:болею|болел)',
                r'(?:мои\s+)?(?:болезни|заболевания)',
            ],
            
            # ПИТОМЦЫ
            FactType.PET_NAME: [
                r'как\s+зовут\s+(?:моего\s+)?(?:кота|кошку|собаку|пса)',
                r'(?:имя|кличка)\s+(?:моего\s+)?(?:питомца|животного)',
            ],
            
            FactType.PET_TYPE: [
                r'(?:какие\s+)?(?:у\s+меня\s+)?(?:питомцы|животные)',
                r'есть\s+ли\s+(?:у\s+меня\s+)?(?:кот|кошка|собака)',
            ],
            
            # НЕДВИЖИМОСТЬ
            FactType.PROPERTY_TYPE: [
                r'(?:где|как)\s+(?:я\s+)?живу',
                r'(?:моя\s+)?(?:квартира|дом|жилье)',
                r'(?:сколько\s+)?комнат',
            ],
            
            # КОНТАКТЫ
            FactType.CONTACT_PHONE: [
                r'(?:мой\s+)?(?:номер|телефон)',
                r'(?:как\s+)?(?:со\s+мной\s+)?(?:связаться|позвонить)',
            ],
            
            FactType.CONTACT_EMAIL: [
                r'(?:мой\s+)?(?:email|почта|мейл)',
                r'(?:электронная\s+)?почта',
            ],
        }
    
    def classify_question(self, question: str) -> Tuple[Optional[FactType], float]:
        """
        Классифицирует вопрос по типу факта
        
        Returns:
            (тип_факта, уверенность)
        """
        question_lower = question.lower().strip()
        
        # Убираем знак вопроса для лучшего матчинга
        question_lower = question_lower.rstrip('?')
        
        best_match = None
        best_confidence = 0.0
        
        for fact_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, question_lower)
                if match:
                    # Рассчитываем уверенность на основе точности совпадения
                    matched_length = len(match.group())
                    total_length = len(question_lower)
                    confidence = matched_length / total_length if total_length > 0 else 0.5
                    
                    # Бонус за точное совпадение
                    if matched_length == total_length:
                        confidence = 1.0
                    
                    if confidence > best_confidence:
                        best_match = fact_type
                        best_confidence = confidence
        
        # Если не нашли точное совпадение, пробуем по ключевым словам
        if best_confidence < 0.5:
            best_match, best_confidence = self._classify_by_keywords(question_lower)
        
        return best_match, best_confidence
    
    def _classify_by_keywords(self, question: str) -> Tuple[Optional[FactType], float]:
        """Классификация по ключевым словам"""
        keyword_map = {
            # Работа
            ('работа', 'профессия', 'должность', 'занимаюсь'): FactType.WORK_OCCUPATION,
            ('компания', 'фирма', 'организация', 'офис'): FactType.WORK_COMPANY,
            ('зарплата', 'доход', 'заработок', 'получаю'): FactType.WORK_SALARY,
            
            # Семья
            ('жена', 'муж', 'супруг', 'женат', 'замужем'): FactType.FAMILY_SPOUSE,
            ('дети', 'сын', 'дочь', 'ребенок'): FactType.FAMILY_CHILDREN,
            ('семья', 'семье', 'родственники'): FactType.FAMILY_STATUS,
            
            # Личное
            ('имя', 'зовут', 'фамилия'): FactType.PERSONAL_NAME,
            ('возраст', 'лет', 'родился'): FactType.PERSONAL_AGE,
            ('живу', 'город', 'адрес'): FactType.PERSONAL_LOCATION,
            
            # Транспорт
            ('машина', 'авто', 'автомобиль', 'тачка'): FactType.TRANSPORT_CAR_BRAND,
            
            # Питомцы
            ('кот', 'кошка', 'собака', 'пес', 'питомец'): FactType.PET_TYPE,
        }
        
        for keywords, fact_type in keyword_map.items():
            for keyword in keywords:
                if keyword in question:
                    # Базовая уверенность для keyword matching
                    return fact_type, 0.4
        
        return None, 0.0
