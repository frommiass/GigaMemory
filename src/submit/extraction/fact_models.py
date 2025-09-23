"""
Модели данных для представления фактов
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import hashlib
import json


class FactType(Enum):
    """Типы фактов о пользователе"""
    # Личная информация
    PERSONAL_NAME = "personal_name"        # Имя, фамилия
    PERSONAL_AGE = "personal_age"         # Возраст
    PERSONAL_LOCATION = "personal_location" # Место жительства
    PERSONAL_GENDER = "personal_gender"    # Пол
    PERSONAL_BIRTH = "personal_birth"      # Дата рождения
    
    # Профессиональная информация
    WORK_OCCUPATION = "work_occupation"    # Профессия
    WORK_COMPANY = "work_company"         # Компания
    WORK_POSITION = "work_position"       # Должность
    WORK_EXPERIENCE = "work_experience"   # Опыт работы
    WORK_SKILLS = "work_skills"          # Навыки
    
    # Семья и отношения
    FAMILY_STATUS = "family_status"       # Семейное положение
    FAMILY_SPOUSE = "family_spouse"       # Супруг(а)
    FAMILY_CHILDREN = "family_children"   # Дети
    FAMILY_PARENTS = "family_parents"     # Родители
    FAMILY_SIBLINGS = "family_siblings"   # Братья/сестры
    FAMILY_RELATIVES = "family_relatives" # Родственники
    
    # Образование
    EDUCATION_LEVEL = "education_level"   # Уровень образования
    EDUCATION_INSTITUTION = "education_institution" # Учебное заведение
    EDUCATION_SPECIALITY = "education_speciality"  # Специальность
    EDUCATION_COURSE = "education_course" # Курсы
    
    # Интересы и хобби
    HOBBY_SPORT = "hobby_sport"          # Спорт
    HOBBY_ACTIVITY = "hobby_activity"    # Активности
    HOBBY_INTEREST = "hobby_interest"    # Интересы
    HOBBY_COLLECTION = "hobby_collection" # Коллекционирование
    
    # Предпочтения
    PREFERENCE_FOOD = "preference_food"   # Еда
    PREFERENCE_MUSIC = "preference_music" # Музыка
    PREFERENCE_MOVIE = "preference_movie" # Фильмы
    PREFERENCE_BOOK = "preference_book"   # Книги
    PREFERENCE_BRAND = "preference_brand" # Бренды
    
    # Здоровье
    HEALTH_CONDITION = "health_condition" # Состояние здоровья
    HEALTH_DISEASE = "health_disease"     # Болезни
    HEALTH_ALLERGY = "health_allergy"    # Аллергии
    HEALTH_MEDICATION = "health_medication" # Лекарства
    
    # Питомцы
    PET_TYPE = "pet_type"                # Тип питомца
    PET_NAME = "pet_name"                # Имя питомца
    PET_BREED = "pet_breed"              # Порода
    
    # События и планы
    EVENT_PAST = "event_past"            # Прошлые события
    EVENT_FUTURE = "event_future"        # Будущие события
    EVENT_TRAVEL = "event_travel"        # Путешествия
    
    # Контакты и социальные сети
    CONTACT_PHONE = "contact_phone"      # Телефон
    CONTACT_EMAIL = "contact_email"      # Email
    CONTACT_SOCIAL = "contact_social"    # Соцсети
    
    # Общее
    GENERAL = "general"                  # Неклассифицированный факт


class FactRelation(Enum):
    """Типы отношений в фактах (предикаты)"""
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
            FactRelation.MARRIED_TO: "в браке с"
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


