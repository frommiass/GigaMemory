"""
Извлекатели фактов из текста - АВТОНОМНАЯ ВЕРСИЯ
Не зависит от внешних модулей, работает изолированно
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
    patterns_matched: int = 0
    conflicts_found: int = 0
    
    def __post_init__(self):
        if self.facts_by_type is None:
            self.facts_by_type = {}
    
    def to_dict(self) -> Dict:
        return {
            'total_extracted': self.total_extracted,
            'facts_by_type': self.facts_by_type,
            'rules_used': self.rules_used,
            'patterns_matched': self.patterns_matched,
            'conflicts_found': self.conflicts_found
        }


class FactExtractor:
    """Базовый класс для извлечения фактов - АВТОНОМНЫЙ"""
    
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
    """
    Извлекатель фактов на основе правил - ПОЛНОСТЬЮ АВТОНОМНЫЙ
    Использует только внутренние паттерны, не зависит от внешних модулей
    """
    
    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
        # Предкомпилируем критические паттерны для скорости
        self._compile_critical_patterns()
    
    def _compile_critical_patterns(self):
        """Предкомпилирует критические паттерны для ускорения"""
        # Критические для конкурса
        self.critical_patterns = {
            'name': FACT_PATTERNS.get(FactType.PERSONAL_NAME, [])[:3],  # Топ-3 паттерна для имени
            'age': FACT_PATTERNS.get(FactType.PERSONAL_AGE, [])[:3],
            'status': FACT_PATTERNS.get(FactType.FAMILY_STATUS, [])[:2],
        }
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя паттерны - БЕЗОПАСНАЯ ВЕРСИЯ"""
        facts = []
        
        # Сначала извлекаем критические факты
        critical_facts = self._extract_critical_facts(text, session_id, dialogue_id)
        facts.extend(critical_facts)
        
        # Затем все остальные с обработкой ошибок
        for fact_type, patterns in FACT_PATTERNS.items():
            try:
                # Пропускаем уже извлеченные критические типы
                if fact_type in [FactType.PERSONAL_NAME, FactType.PERSONAL_AGE, FactType.FAMILY_STATUS]:
                    continue
                
            # Извлекаем все значения по паттернам
            values = extract_all_with_patterns(text, patterns)
            
            for i, value in enumerate(values):
                    try:
                # Нормализуем значение
                normalized_value = normalize_value(value, fact_type)
                
                # Рассчитываем уверенность
                confidence_score = confidence_from_pattern_match(i, len(patterns))
                
                if confidence_score >= self.min_confidence:
                            # Определяем отношение
                    relation = get_relation_for_type(fact_type)
                    
                            # Безопасное получение значения relation
                            if isinstance(relation, FactRelation):
                                relation_value = relation.value
                            else:
                                relation_value = str(relation)
                            
                    # Создаем факт
                    fact = Fact(
                        type=fact_type,
                        subject="пользователь",
                                relation=relation_value,
                        object=normalized_value,
                        confidence=FactConfidence(
                            score=confidence_score,
                            source="rule_based"
                        ),
                        session_id=session_id,
                        dialogue_id=dialogue_id,
                                raw_text=text[:200]  # Сохраняем фрагмент для контекста
                    )
                    
                    facts.append(fact)
                    self.stats.total_extracted += 1
                            self.stats.patterns_matched += 1
                            
                            # Безопасное обновление статистики
                            fact_type_str = fact_type.value if hasattr(fact_type, 'value') else str(fact_type)
                            self.stats.facts_by_type[fact_type_str] = \
                                self.stats.facts_by_type.get(fact_type_str, 0) + 1
                                
                    except Exception as e:
                        logger.debug(f"Failed to process value {value} for type {fact_type}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to process fact type {fact_type}: {e}")
                continue
        
        self.stats.rules_used += 1
        
        # Детектируем временной контекст для фактов
        try:
            temporal_context = detect_temporal_context(text)
            if temporal_context and temporal_context != 'current':
                facts = self._apply_temporal_context(facts, temporal_context)
        except Exception as e:
            logger.debug(f"Temporal context detection failed: {e}")
        
        return facts
    
    def _extract_critical_facts(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает критические факты с повышенной точностью"""
        critical_facts = []
        
        # Извлечение имени - КРИТИЧНО!
        for pattern in self.critical_patterns['name']:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                fact = Fact(
                    type=FactType.PERSONAL_NAME,
                    subject="пользователь",
                    relation=FactRelation.IS.value,
                    object=name.title(),  # Нормализуем имя
                    confidence=FactConfidence(score=0.95, source="critical_pattern"),
                    session_id=session_id,
                    dialogue_id=dialogue_id,
                    raw_text=text[:200]
                )
                critical_facts.append(fact)
                self.stats.total_extracted += 1
                break  # Берем первое найденное имя
        
        # Извлечение возраста
        for pattern in self.critical_patterns['age']:
            match = pattern.search(text)
            if match:
                age = match.group(1)
                try:
                    age_int = int(age)
                    if 0 < age_int < 150:
                        fact = Fact(
                            type=FactType.PERSONAL_AGE,
                            subject="пользователь",
                            relation=FactRelation.IS.value,
                            object=str(age_int),
                            confidence=FactConfidence(score=0.9, source="critical_pattern"),
                            session_id=session_id,
                            dialogue_id=dialogue_id,
                            raw_text=text[:200]
                        )
                        critical_facts.append(fact)
                        self.stats.total_extracted += 1
                        break
                except:
                    continue
        
        return critical_facts
    
    def _apply_temporal_context(self, facts: List[Fact], temporal_context: str) -> List[Fact]:
        """Применяет временной контекст к фактам"""
        temporal_facts = []
        
        for fact in facts:
            # Создаем временной факт для изменяющихся типов
            if fact.type in [FactType.FAMILY_STATUS, FactType.WORK_COMPANY, FactType.PERSONAL_LOCATION]:
                temporal_fact = TemporalFact(
                    type=fact.type,
                    subject=fact.subject,
                    relation=fact.relation,
                    object=fact.object,
                    confidence=fact.confidence,
                    session_id=fact.session_id,
                    dialogue_id=fact.dialogue_id,
                    raw_text=fact.raw_text,
                    is_current=(temporal_context == 'current')
                )
                temporal_facts.append(temporal_fact)
            else:
                temporal_facts.append(fact)
        
        return temporal_facts


class SmartFactExtractor(FactExtractor):
    """
    Умный извлекатель фактов - АВТОНОМНАЯ ВЕРСИЯ
    Работает БЕЗ внешней модели, использует расширенные правила
    """
    
    def __init__(self, model_inference=None, use_rules_first: bool = True, min_confidence: float = 0.6):
        super().__init__()
        self.use_rules_first = use_rules_first
        self.min_confidence = min_confidence
        self.rule_extractor = RuleBasedFactExtractor(min_confidence)
        
        # Если модель не предоставлена, используем только правила
        if model_inference:
            logger.warning("SmartFactExtractor: model_inference ignored in AUTONOMOUS mode")
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя умные правила"""
        all_facts = []
        
        # Используем правила
            rule_facts = self.rule_extractor.extract_facts_from_text(text, session_id, dialogue_id)
            all_facts.extend(rule_facts)
            self.stats.rules_used += 1
        
        # Дополнительная логика для сложных паттернов
        complex_facts = self._extract_complex_patterns(text, session_id, dialogue_id)
        all_facts.extend(complex_facts)
        
        # Обновляем статистику
        self.stats.total_extracted += len(all_facts)
        for fact in all_facts:
            self.stats.facts_by_type[fact.type.value] = \
                self.stats.facts_by_type.get(fact.type.value, 0) + 1
        
        # Убираем дубликаты
        all_facts = self._deduplicate_facts(all_facts)
        
        return all_facts
    
    def _extract_complex_patterns(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты по сложным паттернам"""
        complex_facts = []
        
        # Паттерн: "женился/вышла замуж" -> изменение статуса
        marriage_patterns = [
            r'(?:женился|женюсь|женился недавно|вчера женился)',
            r'(?:вышла замуж|выхожу замуж|замуж вышла)',
            r'(?:поженились|свадьба была|сыграли свадьбу)'
        ]
        
        for pattern in marriage_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                fact = Fact(
                    type=FactType.FAMILY_STATUS,
                    subject="пользователь",
                    relation=FactRelation.IS.value,
                    object="женат" if "женился" in text.lower() else "замужем",
                    confidence=FactConfidence(score=0.9, source="complex_pattern"),
                    session_id=session_id,
                    dialogue_id=dialogue_id,
                    raw_text=text[:200]
                )
                complex_facts.append(fact)
                break
        
        # Паттерн: "развелся/разошлись"
        divorce_patterns = [
            r'(?:развелся|развелась|разводимся)',
            r'(?:расстались|разошлись|больше не вместе)'
        ]
        
        for pattern in divorce_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                fact = Fact(
                    type=FactType.FAMILY_STATUS,
                    subject="пользователь",
                    relation=FactRelation.IS.value,
                    object="разведен",
                    confidence=FactConfidence(score=0.85, source="complex_pattern"),
                    session_id=session_id,
                    dialogue_id=dialogue_id,
                    raw_text=text[:200]
                )
                complex_facts.append(fact)
                break
        
        # Паттерн: смена работы
        job_change_patterns = [
            r'(?:уволился|ушел с работы|больше не работаю)',
            r'(?:перешел в|теперь работаю в|сменил работу)',
            r'(?:новая работа|новое место)'
        ]
        
        for pattern in job_change_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Ищем название новой компании
                company_match = re.search(r'(?:в|на)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)', text[match.end():])
                if company_match:
                    fact = Fact(
                        type=FactType.WORK_COMPANY,
                        subject="пользователь",
                        relation=FactRelation.WORKS_AT.value,
                        object=company_match.group(1),
                        confidence=FactConfidence(score=0.8, source="complex_pattern"),
                        session_id=session_id,
                        dialogue_id=dialogue_id,
                        raw_text=text[:200]
                    )
                    complex_facts.append(fact)
        
        return complex_facts
    
    def _deduplicate_facts(self, facts: List[Fact]) -> List[Fact]:
        """Убирает дубликаты фактов"""
        seen = set()
        unique_facts = []
        
        for fact in facts:
            # Создаем ключ для дедупликации
            key = (fact.type.value, fact.subject, fact.object.lower())
            
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
            else:
                # Если дубликат, повышаем уверенность у существующего
                for existing in unique_facts:
                    if (existing.type == fact.type and 
                        existing.subject == fact.subject and
                        existing.object.lower() == fact.object.lower()):
                        existing.confidence.update(fact.confidence.score)
                        existing.confidence.evidence_count += 1
                        break
        
        return unique_facts


class HybridFactExtractor(FactExtractor):
    """
    Гибридный извлекатель - АВТОНОМНАЯ ВЕРСИЯ
    Комбинирует rule-based и smart подходы БЕЗ внешних зависимостей
    """
    
    def __init__(self, model_inference=None, rule_confidence_threshold: float = 0.7):
        super().__init__()
        self.rule_extractor = RuleBasedFactExtractor()
        self.smart_extractor = SmartFactExtractor(None, use_rules_first=False)
        self.rule_confidence_threshold = rule_confidence_threshold
        
        if model_inference:
            logger.warning("HybridFactExtractor: model_inference ignored in AUTONOMOUS mode")
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Fact]:
        """Извлекает факты используя гибридный подход"""
        # Сначала извлекаем правилами
        rule_facts = self.rule_extractor.extract_facts_from_text(text, session_id, dialogue_id)
        self.stats.rules_used += 1
        
        # Определяем, нужны ли дополнительные методы
        high_confidence_rules = [f for f in rule_facts if f.confidence.score >= self.rule_confidence_threshold]
        
        if len(high_confidence_rules) >= 3:
            # Если правилами извлечено достаточно фактов с высокой уверенностью
            all_facts = rule_facts
        else:
            # Дополняем умным извлечением
            smart_facts = self.smart_extractor._extract_complex_patterns(text, session_id, dialogue_id)
            
            # Объединяем результаты
            all_facts = self._merge_facts(rule_facts, smart_facts)
        
        # Обновляем статистику
        self.stats.total_extracted += len(all_facts)
        for fact in all_facts:
            self.stats.facts_by_type[fact.type.value] = \
                self.stats.facts_by_type.get(fact.type.value, 0) + 1
        
        return all_facts
    
    def _merge_facts(self, rule_facts: List[Fact], smart_facts: List[Fact]) -> List[Fact]:
        """Объединяет факты из разных источников"""
        merged = rule_facts.copy()
        
        for smart_fact in smart_facts:
            # Проверяем, есть ли похожий факт из правил
            is_duplicate = False
            for rule_fact in rule_facts:
                if (smart_fact.type == rule_fact.type and 
                    smart_fact.subject == rule_fact.subject and
                    smart_fact.object.lower() == rule_fact.object.lower()):
                    is_duplicate = True
                    # Обновляем уверенность существующего факта
                    rule_fact.confidence.update(smart_fact.confidence.score)
                    self.stats.conflicts_found += 1
                    break
            
            if not is_duplicate:
                merged.append(smart_fact)
        
        return merged