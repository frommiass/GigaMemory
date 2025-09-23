"""
Модели данных для представления фактов - РАСШИРЕННАЯ ВЕРСИЯ
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import hashlib
import json


class FactType(Enum):
    """Типы фактов о пользователе - РАСШИРЕННАЯ ВЕРСИЯ"""
    
    # === ЛИЧНАЯ ИНФОРМАЦИЯ ===
    PERSONAL_NAME = "personal_name"        # Имя, фамилия
    PERSONAL_NICKNAME = "personal_nickname" # Никнейм, прозвище
    PERSONAL_AGE = "personal_age"         # Возраст
    PERSONAL_BIRTH = "personal_birth"      # Дата рождения
    PERSONAL_LOCATION = "personal_location" # Место жительства
    PERSONAL_HOMETOWN = "personal_hometown" # Родной город
    PERSONAL_DISTRICT = "personal_district" # Район проживания
    PERSONAL_GENDER = "personal_gender"    # Пол
    PERSONAL_NATIONALITY = "personal_nationality" # Национальность
    PERSONAL_LANGUAGE = "personal_language" # Языки
    PERSONAL_HEIGHT = "personal_height"    # Рост
    PERSONAL_WEIGHT = "personal_weight"    # Вес
    PERSONAL_ZODIAC = "personal_zodiac"    # Знак зодиака
    
    # === ПРОФЕССИОНАЛЬНАЯ ИНФОРМАЦИЯ ===
    WORK_OCCUPATION = "work_occupation"    # Профессия
    WORK_COMPANY = "work_company"         # Компания
    WORK_POSITION = "work_position"       # Должность
    WORK_DEPARTMENT = "work_department"   # Отдел
    WORK_EXPERIENCE = "work_experience"   # Опыт работы
    WORK_SKILLS = "work_skills"          # Навыки
    WORK_SALARY = "work_salary"          # Зарплата
    WORK_SCHEDULE = "work_schedule"      # График работы
    WORK_REMOTE = "work_remote"          # Удаленная работа
    WORK_PROJECT = "work_project"        # Текущие проекты
    
    # === СЕМЬЯ И ОТНОШЕНИЯ ===
    FAMILY_STATUS = "family_status"       # Семейное положение
    FAMILY_SPOUSE = "family_spouse"       # Супруг(а)
    FAMILY_CHILDREN = "family_children"   # Дети
    FAMILY_PARENTS = "family_parents"     # Родители
    FAMILY_SIBLINGS = "family_siblings"   # Братья/сестры
    FAMILY_RELATIVES = "family_relatives" # Родственники
    FAMILY_RELATIONSHIP = "family_relationship" # Отношения
    
    # === ОБРАЗОВАНИЕ ===
    EDUCATION_LEVEL = "education_level"   # Уровень образования
    EDUCATION_INSTITUTION = "education_institution" # Учебное заведение
    EDUCATION_SPECIALITY = "education_speciality"  # Специальность
    EDUCATION_DEGREE = "education_degree"  # Степень
    EDUCATION_COURSE = "education_course" # Курсы
    EDUCATION_CERTIFICATION = "education_certification" # Сертификаты
    EDUCATION_YEAR = "education_year"    # Год окончания
    EDUCATION_GPA = "education_gpa"      # Средний балл
    
    # === ТРАНСПОРТ И АВТОМОБИЛИ ===
    TRANSPORT_CAR_BRAND = "transport_car_brand"  # Марка авто
    TRANSPORT_CAR_MODEL = "transport_car_model"  # Модель авто
    TRANSPORT_CAR_YEAR = "transport_car_year"    # Год выпуска
    TRANSPORT_CAR_COLOR = "transport_car_color"  # Цвет авто
    TRANSPORT_CAR_PLATE = "transport_car_plate"  # Номер авто
    TRANSPORT_MOTORCYCLE = "transport_motorcycle" # Мотоцикл
    TRANSPORT_BICYCLE = "transport_bicycle"      # Велосипед
    TRANSPORT_PUBLIC = "transport_public"        # Общественный транспорт
    TRANSPORT_DRIVING_LICENSE = "transport_license" # Водительские права
    TRANSPORT_PARKING = "transport_parking"      # Парковка
    
    # === ФИНАНСЫ И ДЕНЬГИ ===
    FINANCE_INCOME = "finance_income"        # Доход
    FINANCE_SAVINGS = "finance_savings"      # Сбережения
    FINANCE_INVESTMENT = "finance_investment" # Инвестиции
    FINANCE_STOCKS = "finance_stocks"        # Акции
    FINANCE_CRYPTO = "finance_crypto"        # Криптовалюта
    FINANCE_BANK = "finance_bank"           # Банк
    FINANCE_CARD = "finance_card"           # Банковские карты
    FINANCE_CREDIT = "finance_credit"       # Кредит
    FINANCE_MORTGAGE = "finance_mortgage"   # Ипотека
    FINANCE_DEBT = "finance_debt"           # Долги
    FINANCE_BUDGET = "finance_budget"       # Бюджет
    FINANCE_EXPENSES = "finance_expenses"   # Расходы
    FINANCE_INSURANCE = "finance_insurance" # Страховка
    
    # === ЕДА И НАПИТКИ ===
    FOOD_FAVORITE = "food_favorite"         # Любимая еда
    FOOD_DISLIKE = "food_dislike"          # Нелюбимая еда
    FOOD_CUISINE = "food_cuisine"          # Любимая кухня
    FOOD_RESTAURANT = "food_restaurant"    # Любимый ресторан
    FOOD_CAFE = "food_cafe"               # Любимое кафе
    FOOD_DELIVERY = "food_delivery"       # Доставка еды
    FOOD_COOKING = "food_cooking"         # Готовка
    FOOD_DIET = "food_diet"               # Диета
    FOOD_ALLERGY = "food_allergy"         # Пищевая аллергия
    FOOD_VEGETARIAN = "food_vegetarian"   # Вегетарианство
    
    # === НАПИТКИ ===
    DRINK_COFFEE = "drink_coffee"         # Кофе
    DRINK_TEA = "drink_tea"               # Чай
    DRINK_ALCOHOL = "drink_alcohol"       # Алкоголь
    DRINK_BEER = "drink_beer"             # Пиво
    DRINK_WINE = "drink_wine"             # Вино
    DRINK_COCKTAIL = "drink_cocktail"     # Коктейли
    DRINK_JUICE = "drink_juice"           # Соки
    DRINK_SODA = "drink_soda"             # Газировка
    DRINK_WATER = "drink_water"           # Вода
    DRINK_ENERGY = "drink_energy"         # Энергетики
    
    # === ПУТЕШЕСТВИЯ ===
    TRAVEL_COUNTRY = "travel_country"     # Страны
    TRAVEL_CITY = "travel_city"           # Города
    TRAVEL_HOTEL = "travel_hotel"         # Отели
    TRAVEL_FLIGHT = "travel_flight"       # Авиаперелеты
    TRAVEL_VISA = "travel_visa"           # Визы
    TRAVEL_PASSPORT = "travel_passport"   # Паспорт
    TRAVEL_DREAM = "travel_dream"         # Мечты о путешествиях
    TRAVEL_FREQUENCY = "travel_frequency" # Частота путешествий
    TRAVEL_BUDGET = "travel_budget"       # Бюджет путешествий
    TRAVEL_STYLE = "travel_style"         # Стиль путешествий
    
    # === СПОРТ И ФИТНЕС ===
    SPORT_TYPE = "sport_type"             # Вид спорта
    SPORT_TEAM = "sport_team"             # Любимая команда
    SPORT_GYM = "sport_gym"               # Спортзал
    SPORT_TRAINER = "sport_trainer"       # Тренер
    SPORT_ACHIEVEMENT = "sport_achievement" # Достижения
    SPORT_FREQUENCY = "sport_frequency"   # Частота тренировок
    SPORT_EQUIPMENT = "sport_equipment"   # Спортивное снаряжение
    SPORT_NUTRITION = "sport_nutrition"   # Спортивное питание
    SPORT_MARATHON = "sport_marathon"     # Марафоны, забеги
    SPORT_COMPETITION = "sport_competition" # Соревнования
    
    # === ХОББИ И ИНТЕРЕСЫ ===
    HOBBY_ACTIVITY = "hobby_activity"     # Активности
    HOBBY_COLLECTION = "hobby_collection" # Коллекционирование
    HOBBY_GAMING = "hobby_gaming"         # Игры
    HOBBY_READING = "hobby_reading"       # Чтение
    HOBBY_MUSIC = "hobby_music"           # Музыка
    HOBBY_INSTRUMENT = "hobby_instrument" # Музыкальные инструменты
    HOBBY_PHOTOGRAPHY = "hobby_photography" # Фотография
    HOBBY_ART = "hobby_art"               # Искусство
    HOBBY_CRAFTS = "hobby_crafts"         # Рукоделие
    HOBBY_GARDENING = "hobby_gardening"   # Садоводство
    HOBBY_FISHING = "hobby_fishing"       # Рыбалка
    HOBBY_HUNTING = "hobby_hunting"       # Охота
    
    # === ТЕХНОЛОГИИ ===
    TECH_PHONE = "tech_phone"             # Телефон
    TECH_LAPTOP = "tech_laptop"           # Ноутбук
    TECH_TABLET = "tech_tablet"           # Планшет
    TECH_SMARTWATCH = "tech_smartwatch"   # Умные часы
    TECH_GAMING_CONSOLE = "tech_console"  # Игровая консоль
    TECH_SMART_HOME = "tech_smart_home"   # Умный дом
    TECH_PROGRAMMING = "tech_programming" # Языки программирования
    TECH_SOFTWARE = "tech_software"       # Используемое ПО
    
    # === ПРЕДПОЧТЕНИЯ И ВКУСЫ ===
    PREFERENCE_COLOR = "preference_color"  # Любимый цвет
    PREFERENCE_SEASON = "preference_season" # Любимое время года
    PREFERENCE_WEATHER = "preference_weather" # Любимая погода
    PREFERENCE_MOVIE = "preference_movie"  # Фильмы
    PREFERENCE_SERIES = "preference_series" # Сериалы
    PREFERENCE_BOOK = "preference_book"    # Книги
    PREFERENCE_AUTHOR = "preference_author" # Авторы
    PREFERENCE_MUSIC_GENRE = "preference_music_genre" # Жанр музыки
    PREFERENCE_ARTIST = "preference_artist" # Исполнители
    PREFERENCE_BRAND = "preference_brand"  # Бренды
    PREFERENCE_SHOP = "preference_shop"    # Магазины
    
    # === ЗДОРОВЬЕ ===
    HEALTH_CONDITION = "health_condition"  # Состояние здоровья
    HEALTH_DISEASE = "health_disease"      # Болезни
    HEALTH_ALLERGY = "health_allergy"     # Аллергии
    HEALTH_MEDICATION = "health_medication" # Лекарства
    HEALTH_DOCTOR = "health_doctor"       # Врачи
    HEALTH_CLINIC = "health_clinic"       # Клиники
    HEALTH_INSURANCE = "health_insurance" # Медстраховка
    HEALTH_BLOOD_TYPE = "health_blood_type" # Группа крови
    HEALTH_VACCINE = "health_vaccine"     # Прививки
    HEALTH_MENTAL = "health_mental"       # Ментальное здоровье
    
    # === ПИТОМЦЫ ===
    PET_TYPE = "pet_type"                 # Тип питомца
    PET_NAME = "pet_name"                 # Имя питомца
    PET_BREED = "pet_breed"               # Порода
    PET_AGE = "pet_age"                   # Возраст питомца
    PET_VET = "pet_vet"                   # Ветеринар
    PET_FOOD = "pet_food"                 # Корм
    
    # === НЕДВИЖИМОСТЬ ===
    PROPERTY_TYPE = "property_type"       # Тип жилья
    PROPERTY_OWNERSHIP = "property_ownership" # Собственность
    PROPERTY_ROOMS = "property_rooms"     # Количество комнат
    PROPERTY_AREA = "property_area"       # Площадь
    PROPERTY_FLOOR = "property_floor"     # Этаж
    PROPERTY_RENOVATION = "property_renovation" # Ремонт
    
    # === СОБЫТИЯ И ПЛАНЫ ===
    EVENT_BIRTHDAY = "event_birthday"     # День рождения
    EVENT_ANNIVERSARY = "event_anniversary" # Годовщины
    EVENT_WEDDING = "event_wedding"       # Свадьба
    EVENT_VACATION = "event_vacation"     # Отпуск
    EVENT_MEETING = "event_meeting"       # Встречи
    EVENT_PARTY = "event_party"           # Вечеринки
    EVENT_CONCERT = "event_concert"       # Концерты
    EVENT_FUTURE_PLAN = "event_future"    # Планы на будущее
    
    # === КОНТАКТЫ И СОЦИАЛЬНЫЕ СЕТИ ===
    CONTACT_PHONE = "contact_phone"       # Телефон
    CONTACT_EMAIL = "contact_email"       # Email
    CONTACT_TELEGRAM = "contact_telegram" # Telegram
    CONTACT_WHATSAPP = "contact_whatsapp" # WhatsApp
    CONTACT_INSTAGRAM = "contact_instagram" # Instagram
    CONTACT_VK = "contact_vk"             # VKontakte
    CONTACT_FACEBOOK = "contact_facebook" # Facebook
    CONTACT_LINKEDIN = "contact_linkedin" # LinkedIn
    CONTACT_TWITTER = "contact_twitter"   # Twitter
    CONTACT_TIKTOK = "contact_tiktok"     # TikTok
    
    # === ОБЩЕЕ ===
    GENERAL = "general"                   # Неклассифицированный факт


class FactRelation(Enum):
    """Типы отношений в фактах (предикаты) - РАСШИРЕННАЯ ВЕРСИЯ"""
    # Базовые отношения
    IS = "is"                    # является
    HAS = "has"                  # имеет
    WORKS_AS = "works_as"        # работает как
    WORKS_AT = "works_at"        # работает в
    LIVES_IN = "lives_in"        # живет в
    BORN_IN = "born_in"          # родился в
    STUDIED_AT = "studied_at"    # учился в
    LIKES = "likes"              # любит
    DISLIKES = "dislikes"        # не любит
    OWNS = "owns"                # владеет
    MARRIED_TO = "married_to"    # женат/замужем за
    PARENT_OF = "parent_of"      # родитель
    CHILD_OF = "child_of"        # ребенок
    SIBLING_OF = "sibling_of"    # брат/сестра
    
    # Финансовые отношения
    EARNS = "earns"              # зарабатывает
    SAVES = "saves"              # сберегает
    INVESTS_IN = "invests_in"    # инвестирует в
    OWES = "owes"                # должен
    BANKS_WITH = "banks_with"    # обслуживается в банке
    
    # Предпочтения
    PREFERS = "prefers"          # предпочитает
    DRINKS = "drinks"            # пьет
    EATS = "eats"                # ест
    DRIVES = "drives"            # водит
    
    # Временные отношения
    WAS = "was"                  # был
    WILL_BE = "will_be"          # будет
    USED_TO = "used_to"          # раньше
    PLANS_TO = "plans_to"        # планирует
    
    # Действия
    DOES = "does"                # делает
    PLAYS = "plays"              # играет
    VISITS = "visits"            # посещает
    TRAVELS_TO = "travels_to"    # путешествует в
    TRAINS_AT = "trains_at"      # тренируется в
    SHOPS_AT = "shops_at"        # покупает в
    COOKS = "cooks"              # готовит
    COLLECTS = "collects"        # коллекционирует
    WATCHES = "watches"          # смотрит
    READS = "reads"              # читает
    LISTENS_TO = "listens_to"    # слушает
    
    # Общее
    RELATES_TO = "relates_to"    # относится к


@dataclass
class FactConfidence:
    """Уровень уверенности в факте"""
    score: float  # от 0.0 до 1.0
    source: str   # источник (direct, inferred, extracted)
    evidence_count: int = 1  # количество подтверждений
    
    def __post_init__(self):
        """Валидация уверенности"""
        self.score = max(0.0, min(1.0, self.score))
    
    @property
    def level(self) -> str:
        """Уровень уверенности текстом"""
        if self.score >= 0.9:
            return "очень высокая"
        elif self.score >= 0.7:
            return "высокая"
        elif self.score >= 0.5:
            return "средняя"
        elif self.score >= 0.3:
            return "низкая"
        else:
            return "очень низкая"
    
    def update(self, new_score: float, increase_evidence: bool = True):
        """Обновляет уверенность с учетом нового свидетельства"""
        if increase_evidence:
            self.evidence_count += 1
        # Взвешенное среднее с учетом количества свидетельств
        self.score = (self.score * (self.evidence_count - 1) + new_score) / self.evidence_count
        self.score = max(0.0, min(1.0, self.score))


@dataclass
class Fact:
    """Базовый класс для представления факта"""
    # Основные поля
    type: FactType                    # Тип факта
    subject: str                      # Субъект (обычно "пользователь" или имя)
    relation: Union[FactRelation, str]  # Отношение/предикат
    object: str                       # Объект факта (значение)
    
    # Метаданные
    confidence: FactConfidence        # Уверенность в факте
    session_id: str                   # ID сессии, откуда извлечен факт
    dialogue_id: str                  # ID диалога
    
    # Дополнительно
    raw_text: Optional[str] = None   # Исходный текст
    extracted_at: datetime = field(default_factory=datetime.now)  # Время извлечения
    attributes: Dict[str, Any] = field(default_factory=dict)  # Дополнительные атрибуты
    
    def __post_init__(self):
        """Постобработка после создания"""
        # Преобразуем строку в FactRelation если нужно
        if isinstance(self.relation, str):
            try:
                self.relation = FactRelation(self.relation)
            except ValueError:
                # Если не стандартное отношение, оставляем как строку
                pass
        
        # Генерируем ID факта
        self.id = self.generate_id()
    
    def generate_id(self) -> str:
        """Генерирует уникальный ID факта на основе содержимого"""
        content = f"{self.type.value}:{self.subject}:{self.relation}:{self.object}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует факт в словарь"""
        return {
            'id': self.id,
            'type': self.type.value,
            'subject': self.subject,
            'relation': self.relation.value if isinstance(self.relation, FactRelation) else self.relation,
            'object': self.object,
            'confidence': {
                'score': self.confidence.score,
                'level': self.confidence.level,
                'source': self.confidence.source,
                'evidence_count': self.confidence.evidence_count
            },
            'session_id': self.session_id,
            'dialogue_id': self.dialogue_id,
            'raw_text': self.raw_text[:200] if self.raw_text else None,
            'extracted_at': self.extracted_at.isoformat(),
            'attributes': self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """Создает факт из словаря"""
        confidence = FactConfidence(
            score=data['confidence']['score'],
            source=data['confidence']['source'],
            evidence_count=data['confidence'].get('evidence_count', 1)
        )
        
        fact = cls(
            type=FactType(data['type']),
            subject=data['subject'],
            relation=data['relation'],
            object=data['object'],
            confidence=confidence,
            session_id=data['session_id'],
            dialogue_id=data['dialogue_id'],
            raw_text=data.get('raw_text'),
            attributes=data.get('attributes', {})
        )
        
        if 'extracted_at' in data:
            fact.extracted_at = datetime.fromisoformat(data['extracted_at'])
        
        return fact
    
    def to_natural_text(self) -> str:
        """Преобразует факт в естественный текст"""
        relation_text = {
            FactRelation.IS: "это",
            FactRelation.HAS: "имеет",
            FactRelation.WORKS_AS: "работает",
            FactRelation.WORKS_AT: "работает в",
            FactRelation.LIVES_IN: "живет в",
            FactRelation.LIKES: "любит",
            FactRelation.OWNS: "владеет",
            FactRelation.MARRIED_TO: "в браке с",
            FactRelation.EARNS: "зарабатывает",
            FactRelation.DRIVES: "водит",
            FactRelation.DRINKS: "пьет",
            FactRelation.EATS: "ест",
            FactRelation.TRAINS_AT: "тренируется в",
            FactRelation.INVESTS_IN: "инвестирует в",
        }
        
        rel_text = relation_text.get(self.relation, str(self.relation))
        return f"{self.subject} {rel_text} {self.object}"
    
    def is_conflicting_with(self, other: 'Fact') -> bool:
        """Проверяет, конфликтует ли факт с другим"""
        # Факты конфликтуют если у них одинаковые subject, type и relation,
        # но разные object
        if (self.subject == other.subject and 
            self.type == other.type and 
            self.relation == other.relation and
            self.object != other.object):
            return True
        
        # Специальные случаи конфликтов
        # Например, возраст не может быть двумя разными числами
        if (self.type == FactType.PERSONAL_AGE and 
            other.type == FactType.PERSONAL_AGE and
            self.subject == other.subject and
            self.object != other.object):
            return True
        
        return False
    
    def __str__(self) -> str:
        """Строковое представление"""
        return f"[{self.type.value}] {self.to_natural_text()} (confidence: {self.confidence.score:.2f})"
    
    def __repr__(self) -> str:
        """Отладочное представление"""
        return f"Fact(id={self.id}, type={self.type.value}, object={self.object})"


@dataclass
class TemporalFact(Fact):
    """Факт с временной меткой"""
    timestamp: Optional[datetime] = None  # Когда факт был актуален
    valid_from: Optional[datetime] = None  # С какого времени действует
    valid_until: Optional[datetime] = None  # До какого времени действует
    is_current: bool = True  # Актуален ли сейчас
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Проверяет, действителен ли факт в указанное время"""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует временной факт в словарь"""
        data = super().to_dict()
        data['temporal'] = {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'is_current': self.is_current
        }
        return data


@dataclass
class ConflictingFacts:
    """Контейнер для конфликтующих фактов"""
    facts: List[Fact]
    resolved_fact: Optional[Fact] = None
    resolution_reason: Optional[str] = None
    
    def add_fact(self, fact: Fact):
        """Добавляет факт в список конфликтующих"""
        self.facts.append(fact)
    
    def resolve(self, strategy: str = "latest") -> Fact:
        """
        Разрешает конфликт между фактами
        
        Args:
            strategy: Стратегия разрешения
                - "latest": Берем самый поздний факт
                - "highest_confidence": Берем с наибольшей уверенностью
                - "most_evidence": Берем с наибольшим количеством свидетельств
        """
        if not self.facts:
            return None
        
        if strategy == "latest":
            # Сортируем по session_id (предполагаем, что больший ID = позже)
            resolved = max(self.facts, key=lambda f: int(f.session_id) if f.session_id.isdigit() else 0)
            self.resolution_reason = "выбран более поздний факт"
        
        elif strategy == "highest_confidence":
            # Выбираем с максимальной уверенностью
            resolved = max(self.facts, key=lambda f: f.confidence.score)
            self.resolution_reason = f"выбран факт с уверенностью {resolved.confidence.score:.2f}"
        
        elif strategy == "most_evidence":
            # Выбираем с наибольшим количеством свидетельств
            resolved = max(self.facts, key=lambda f: f.confidence.evidence_count)
            self.resolution_reason = f"выбран факт с {resolved.confidence.evidence_count} свидетельствами"
        
        else:
            # По умолчанию берем первый
            resolved = self.facts[0]
            self.resolution_reason = "стратегия не определена, выбран первый факт"
        
        self.resolved_fact = resolved
        return resolved
    
    def get_all_values(self) -> List[str]:
        """Возвращает все конфликтующие значения"""
        return [fact.object for fact in self.facts]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            'conflict_type': self.facts[0].type.value if self.facts else None,
            'conflicting_values': self.get_all_values(),
            'facts': [f.to_dict() for f in self.facts],
            'resolved': self.resolved_fact.to_dict() if self.resolved_fact else None,
            'resolution_reason': self.resolution_reason
        }