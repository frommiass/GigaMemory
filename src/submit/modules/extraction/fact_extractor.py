"""
Извлекатели фактов из текста - УЛУЧШЕННАЯ ВЕРСИЯ
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .fact_models import Fact, FactType, FactRelation, FactConfidence, TemporalFact
from .fact_patterns import (
    FACT_PATTERNS, extract_with_pattern, extract_all_with_patterns,
    detect_temporal_context, normalize_value, confidence_from_pattern_match,
    get_relation_for_type
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Статистика извлечения фактов"""
    total_extracted: int = 0
    facts_by_type: Dict[str, int] = None
    rules_used: int = 0
    llm_used: int = 0
    conflicts_found: int = 0
    
    def __post_init__(self):
        if self.facts_by_type is None:
            self.facts_by_type = {}
    
    def to_dict(self) -> Dict:
        return {
            'total_extracted': self.total_extracted,
            'facts_by_type': self.facts_by_type,
            'rules_used': self.rules_used,
            'llm_used': self.llm_used,
            'conflicts_found': self.conflicts_found
        }


class FactExtractor:
    """Базовый класс для извлечения фактов"""
    
    def __init__(self):
        self.stats = ExtractionStats()
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """
        Извлекает факты из текста
        
        Args:
            text: Исходный текст
            session_id: ID сессии
            dialogue_id: ID диалога
            
        Returns:
            Список извлеченных фактов
        """
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """Возвращает статистику извлечения"""
        return self.stats.to_dict()


class RuleBasedFactExtractor(FactExtractor):
    """Извлекатель фактов на основе правил (регулярных выражений) - УЛУЧШЕННАЯ ВЕРСИЯ"""
    
    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя паттерны"""
        facts = []
        
        for fact_type, patterns in FACT_PATTERNS.items():
            # Извлекаем все значения по паттернам
            values = extract_all_with_patterns(text, patterns)
            
            for i, value in enumerate(values):
                # Нормализуем значение
                normalized_value = normalize_value(value, fact_type)
                
                # Рассчитываем уверенность
                confidence_score = confidence_from_pattern_match(i, len(patterns))
                
                if confidence_score >= self.min_confidence:
                    # Определяем отношение используя новую функцию
                    relation = get_relation_for_type(fact_type)
                    
                    # Создаем факт
                    fact = Fact(
                        type=fact_type,
                        subject="пользователь",
                        relation=relation.value,  # Преобразуем enum в строку
                        object=normalized_value,
                        confidence=FactConfidence(
                            score=confidence_score,
                            source="rule_based"
                        ),
                        session_id=session_id,
                        dialogue_id=dialogue_id,
                        raw_text=text
                    )
                    
                    facts.append(fact)
                    self.stats.total_extracted += 1
                    self.stats.facts_by_type[fact_type.value] = \
                        self.stats.facts_by_type.get(fact_type.value, 0) + 1
        
        self.stats.rules_used += 1
        return facts


class SmartFactExtractor(FactExtractor):
    """Умный извлекатель фактов с использованием LLM - УЛУЧШЕННАЯ ВЕРСИЯ"""
    
    def __init__(self, model_inference, use_rules_first: bool = True):
        super().__init__()
        self.model_inference = model_inference
        self.use_rules_first = use_rules_first
        self.rule_extractor = RuleBasedFactExtractor() if use_rules_first else None
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя LLM и правила"""
        all_facts = []
        
        # Сначала используем правила если включено
        if self.use_rules_first and self.rule_extractor:
            rule_facts = self.rule_extractor.extract_facts_from_text(text, session_id, dialogue_id)
            all_facts.extend(rule_facts)
            self.stats.rules_used += 1
        
        # Затем используем LLM (если доступен)
        if self.model_inference:
            llm_facts = self._extract_with_llm(text, session_id, dialogue_id)
            all_facts.extend(llm_facts)
            self.stats.llm_used += 1
        
        # Обновляем статистику
        self.stats.total_extracted += len(all_facts)
        for fact in all_facts:
            self.stats.facts_by_type[fact.type.value] = \
                self.stats.facts_by_type.get(fact.type.value, 0) + 1
        
        return all_facts
    
    def _extract_with_llm(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя LLM"""
        try:
            # Если модель не доступна, возвращаем пустой список
            if not self.model_inference:
                return []
            
            # Создаем промпт для извлечения фактов
            prompt = self._create_extraction_prompt(text)
            
            # Вызываем модель
            response = self.model_inference.inference([{"role": "user", "content": prompt}])
            
            # Парсим JSON ответ
            facts_data = self._parse_llm_response(response)
            
            # Создаем факты
            facts = []
            for fact_data in facts_data:
                try:
                    fact = self._create_fact_from_llm_data(fact_data, text, session_id, dialogue_id)
                    if fact:
                        facts.append(fact)
                except Exception as e:
                    logger.warning(f"Ошибка создания факта из LLM данных: {e}")
                    continue
            
            return facts
            
        except Exception as e:
            logger.error(f"Ошибка извлечения фактов с LLM: {e}")
            return []
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Создает промпт для извлечения фактов - РАСШИРЕННАЯ ВЕРСИЯ"""
        return f"""
Извлеки структурированные факты из следующего текста на русском языке.

Текст: "{text}"

Верни результат в формате JSON массива. Каждый факт должен содержать:
- type: тип факта (см. список ниже)
- subject: субъект (обычно "пользователь")
- relation: отношение (is, has, works_as, works_at, lives_in, married_to, owns, does, likes, studied_at, travels_to, earns, drinks, drives, invests_in, trains_at, prefers)
- object: значение факта
- confidence: уверенность от 0.0 до 1.0

Основные типы фактов:
ЛИЧНАЯ ИНФОРМАЦИЯ: personal_name, personal_nickname, personal_age, personal_birth, personal_location, personal_hometown, personal_district, personal_gender, personal_nationality, personal_language

РАБОТА: work_occupation, work_company, work_position, work_department, work_salary, work_experience, work_skills, work_schedule, work_project

ТРАНСПОРТ: transport_car_brand, transport_car_model, transport_car_year, transport_car_color, transport_motorcycle, transport_bicycle, transport_license

ФИНАНСЫ: finance_income, finance_savings, finance_investment, finance_stocks, finance_crypto, finance_bank, finance_card, finance_credit, finance_mortgage, finance_debt, finance_budget

ЕДА И НАПИТКИ: food_favorite, food_dislike, food_cuisine, food_restaurant, food_cafe, food_delivery, food_diet, drink_coffee, drink_tea, drink_alcohol, drink_beer, drink_wine

ПУТЕШЕСТВИЯ: travel_country, travel_city, travel_hotel, travel_dream, travel_frequency, travel_budget

СПОРТ: sport_type, sport_team, sport_gym, sport_trainer, sport_frequency, sport_equipment, sport_achievement

СЕМЬЯ: family_status, family_spouse, family_children, family_parents, family_siblings

ОБРАЗОВАНИЕ: education_institution, education_speciality, education_degree, education_course, education_year

ХОББИ: hobby_activity, hobby_gaming, hobby_reading, hobby_music, hobby_instrument, hobby_photography, hobby_art

ТЕХНОЛОГИИ: tech_phone, tech_laptop, tech_programming, tech_software

ЗДОРОВЬЕ: health_condition, health_disease, health_allergy, health_medication, health_doctor

ПИТОМЦЫ: pet_type, pet_name, pet_breed, pet_age

НЕДВИЖИМОСТЬ: property_type, property_ownership, property_rooms, property_area

КОНТАКТЫ: contact_phone, contact_email, contact_telegram, contact_instagram

Пример:
[
  {{"type": "personal_name", "subject": "пользователь", "relation": "is", "object": "Иван", "confidence": 0.9}},
  {{"type": "work_occupation", "subject": "пользователь", "relation": "works_as", "object": "программист", "confidence": 0.8}},
  {{"type": "transport_car_brand", "subject": "пользователь", "relation": "drives", "object": "Toyota", "confidence": 0.85}},
  {{"type": "drink_coffee", "subject": "пользователь", "relation": "drinks", "object": "капучино", "confidence": 0.75}},
  {{"type": "finance_bank", "subject": "пользователь", "relation": "banks_with", "object": "Тинькофф", "confidence": 0.8}}
]

Извлеки только достоверные факты. Если факт неясен или сомнителен, не включай его.
Обращай внимание на детали и извлекай максимум информации из текста.
"""
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Парсит ответ LLM в список фактов"""
        try:
            # Очищаем ответ от лишнего текста
            response = response.strip()
            
            # Ищем JSON в ответе
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            return []
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга ответа LLM: {e}")
            return []
    
    def _create_fact_from_llm_data(self, data: Dict, text: str, session_id: str, dialogue_id: str) -> Optional[Fact]:
        """Создает факт из данных LLM"""
        try:
            # Валидируем обязательные поля
            required_fields = ['type', 'subject', 'relation', 'object', 'confidence']
            if not all(field in data for field in required_fields):
                return None
            
            # Преобразуем тип факта
            fact_type = FactType(data['type'])
            
            # Преобразуем отношение
            relation = FactRelation(data['relation'])
            
            # Создаем уверенность
            confidence = FactConfidence(
                score=float(data['confidence']),
                source="llm_extraction"
            )
            
            # Создаем факт
            fact = Fact(
                type=fact_type,
                subject=data['subject'],
                relation=relation,
                object=str(data['object']),
                confidence=confidence,
                session_id=session_id,
                dialogue_id=dialogue_id,
                raw_text=text
            )
            
            return fact
            
        except Exception as e:
            logger.warning(f"Ошибка создания факта: {e}")
            return None


class HybridFactExtractor(FactExtractor):
    """Гибридный извлекатель, комбинирующий правила и LLM - УЛУЧШЕННАЯ ВЕРСИЯ"""
    
    def __init__(self, model_inference, rule_confidence_threshold: float = 0.7):
        super().__init__()
        self.model_inference = model_inference
        self.rule_extractor = RuleBasedFactExtractor()
        self.smart_extractor = SmartFactExtractor(model_inference, use_rules_first=False)
        self.rule_confidence_threshold = rule_confidence_threshold
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя гибридный подход"""
        # Сначала извлекаем правилами
        rule_facts = self.rule_extractor.extract_facts_from_text(text, session_id, dialogue_id)
        self.stats.rules_used += 1
        
        # Определяем, нужен ли LLM
        high_confidence_rules = [f for f in rule_facts if f.confidence.score >= self.rule_confidence_threshold]
        
        if len(high_confidence_rules) >= 3:
            # Если правилами извлечено достаточно фактов с высокой уверенностью, используем только их
            all_facts = rule_facts
        else:
            # Иначе дополняем LLM
            llm_facts = self.smart_extractor._extract_with_llm(text, session_id, dialogue_id)
            self.stats.llm_used += 1
            
            # Объединяем результаты, избегая дубликатов
            all_facts = self._merge_facts(rule_facts, llm_facts)
        
        # Обновляем статистику
        self.stats.total_extracted += len(all_facts)
        for fact in all_facts:
            self.stats.facts_by_type[fact.type.value] = \
                self.stats.facts_by_type.get(fact.type.value, 0) + 1
        
        return all_facts
    
    def _merge_facts(self, rule_facts: List[Fact], llm_facts: List[Fact]) -> List[Fact]:
        """Объединяет факты из правил и LLM, избегая дубликатов"""
        merged = rule_facts.copy()
        
        for llm_fact in llm_facts:
            # Проверяем, есть ли похожий факт из правил
            is_duplicate = False
            for rule_fact in rule_facts:
                if (llm_fact.type == rule_fact.type and 
                    llm_fact.subject == rule_fact.subject and
                    llm_fact.object.lower() == rule_fact.object.lower()):
                    is_duplicate = True
                    # Обновляем уверенность существующего факта
                    rule_fact.confidence.update(llm_fact.confidence.score)
                    break
            
            if not is_duplicate:
                merged.append(llm_fact)
        
        return merged


class AdvancedFactExtractor(FactExtractor):
    """Продвинутый извлекатель с дополнительными возможностями"""
    
    def __init__(self, model_inference=None, use_context: bool = True, 
                 context_window: int = 3, min_confidence: float = 0.5):
        super().__init__()
        self.model_inference = model_inference
        self.use_context = use_context
        self.context_window = context_window
        self.min_confidence = min_confidence
        self.rule_extractor = RuleBasedFactExtractor(min_confidence)
        
        # Кэш для контекста
        self.context_cache = []
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты с учетом контекста"""
        facts = []
        
        # Добавляем текст в контекст
        if self.use_context:
            self.context_cache.append(text)
            if len(self.context_cache) > self.context_window:
                self.context_cache.pop(0)
        
        # Извлекаем базовые факты
        facts = self.rule_extractor.extract_facts_from_text(text, session_id, dialogue_id)
        
        # Анализируем контекст для повышения уверенности
        if self.use_context and len(self.context_cache) > 1:
            facts = self._enhance_with_context(facts)
        
        # Извлекаем составные факты
        composite_facts = self._extract_composite_facts(text, facts, session_id, dialogue_id)
        facts.extend(composite_facts)
        
        # Обновляем статистику
        self.stats.total_extracted += len(facts)
        for fact in facts:
            self.stats.facts_by_type[fact.type.value] = \
                self.stats.facts_by_type.get(fact.type.value, 0) + 1
        
        return facts
    
    def _enhance_with_context(self, facts: List[Fact]) -> List[Fact]:
        """Улучшает факты используя контекст"""
        enhanced_facts = []
        
        for fact in facts:
            # Проверяем, упоминался ли факт в контексте
            context_mentions = 0
            for context_text in self.context_cache:
                if fact.object.lower() in context_text.lower():
                    context_mentions += 1
            
            # Увеличиваем уверенность если факт упоминался ранее
            if context_mentions > 1:
                fact.confidence.update(
                    min(1.0, fact.confidence.score + 0.1 * context_mentions)
                )
            
            enhanced_facts.append(fact)
        
        return enhanced_facts
    
    def _extract_composite_facts(self, text: str, base_facts: List[Fact], 
                                session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает составные факты на основе базовых"""
        composite_facts = []
        
        # Пример: если есть марка и модель авто, создаем составной факт
        car_brand = None
        car_model = None
        
        for fact in base_facts:
            if fact.type == FactType.TRANSPORT_CAR_BRAND:
                car_brand = fact.object
            elif fact.type == FactType.TRANSPORT_CAR_MODEL:
                car_model = fact.object
        
        if car_brand and car_model:
            composite = Fact(
                type=FactType.TRANSPORT_CAR_BRAND,
                subject="пользователь",
                relation=FactRelation.DRIVES,
                object=f"{car_brand} {car_model}",
                confidence=FactConfidence(score=0.9, source="composite"),
                session_id=session_id,
                dialogue_id=dialogue_id,
                raw_text=text
            )
            composite_facts.append(composite)
        
        return composite_facts