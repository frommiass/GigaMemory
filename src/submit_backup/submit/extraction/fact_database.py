"""
База данных для хранения и управления фактами
"""
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from .fact_models import (
    Fact, FactType, FactRelation, FactConfidence,
    TemporalFact, ConflictingFacts
)

logger = logging.getLogger(__name__)


@dataclass
class FactStats:
    """Статистика по фактам"""
    total_facts: int = 0
    facts_by_type: Dict[str, int] = field(default_factory=dict)
    facts_by_dialogue: Dict[str, int] = field(default_factory=dict)
    conflicts_resolved: int = 0
    temporal_facts: int = 0
    average_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Преобразует статистику в словарь"""
        return {
            'total_facts': self.total_facts,
            'facts_by_type': self.facts_by_type,
            'facts_by_dialogue': self.facts_by_dialogue,
            'conflicts_resolved': self.conflicts_resolved,
            'temporal_facts': self.temporal_facts,
            'average_confidence': self.average_confidence
        }


class FactIndex:
    """Индекс для быстрого поиска фактов"""
    
    def __init__(self):
        # Индексы по разным критериям
        self.by_type: Dict[FactType, List[str]] = defaultdict(list)
        self.by_subject: Dict[str, List[str]] = defaultdict(list)
        self.by_relation: Dict[FactRelation, List[str]] = defaultdict(list)
        self.by_dialogue: Dict[str, List[str]] = defaultdict(list)
        self.by_session: Dict[str, List[str]] = defaultdict(list)
        
        # Индекс для поиска по объекту (значению)
        self.by_object: Dict[str, List[str]] = defaultdict(list)
        
        # Полнотекстовый индекс (простой)
        self.text_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_fact(self, fact: Fact):
        """Индексирует факт"""
        fact_id = fact.id
        
        # Добавляем в индексы
        self.by_type[fact.type].append(fact_id)
        self.by_subject[fact.subject].append(fact_id)
        if isinstance(fact.relation, FactRelation):
            self.by_relation[fact.relation].append(fact_id)
        self.by_dialogue[fact.dialogue_id].append(fact_id)
        self.by_session[fact.session_id].append(fact_id)
        
        # Индексируем объект
        object_lower = fact.object.lower()
        self.by_object[object_lower].append(fact_id)
        
        # Добавляем в текстовый индекс
        self._update_text_index(fact)
    
    def remove_fact(self, fact: Fact):
        """Удаляет факт из индексов"""
        fact_id = fact.id
        
        # Удаляем из всех индексов
        self._remove_from_list(self.by_type[fact.type], fact_id)
        self._remove_from_list(self.by_subject[fact.subject], fact_id)
        if isinstance(fact.relation, FactRelation):
            self._remove_from_list(self.by_relation[fact.relation], fact_id)
        self._remove_from_list(self.by_dialogue[fact.dialogue_id], fact_id)
        self._remove_from_list(self.by_session[fact.session_id], fact_id)
        
        object_lower = fact.object.lower()
        self._remove_from_list(self.by_object[object_lower], fact_id)
        
        # Удаляем из текстового индекса
        self._remove_from_text_index(fact)
    
    def _update_text_index(self, fact: Fact):
        """Обновляет текстовый индекс"""
        # Извлекаем слова из объекта факта
        words = fact.object.lower().split()
        for word in words:
            if len(word) > 2:  # Индексируем слова длиннее 2 символов
                self.text_index[word].add(fact.id)
        
        # Также индексируем слова из raw_text если есть
        if fact.raw_text:
            words = fact.raw_text.lower().split()[:20]  # Первые 20 слов
            for word in words:
                if len(word) > 2:
                    self.text_index[word].add(fact.id)
    
    def _remove_from_text_index(self, fact: Fact):
        """Удаляет факт из текстового индекса"""
        # Удаляем из всех множеств где есть fact.id
        for word_set in self.text_index.values():
            word_set.discard(fact.id)
    
    def _remove_from_list(self, lst: List, item):
        """Безопасно удаляет элемент из списка"""
        try:
            lst.remove(item)
        except ValueError:
            pass
    
    def search_by_text(self, query: str) -> Set[str]:
        """Полнотекстовый поиск"""
        query_words = query.lower().split()
        result_ids = set()
        
        for word in query_words:
            if word in self.text_index:
                result_ids.update(self.text_index[word])
        
        return result_ids
    
    def clear(self):
        """Очищает все индексы"""
        self.by_type.clear()
        self.by_subject.clear()
        self.by_relation.clear()
        self.by_dialogue.clear()
        self.by_session.clear()
        self.by_object.clear()
        self.text_index.clear()


class FactConflictResolver:
    """Класс для разрешения конфликтов между фактами"""
    
    def __init__(self, strategy: str = "latest"):
        """
        Инициализация резолвера конфликтов
        
        Args:
            strategy: Стратегия разрешения конфликтов
                - "latest": Берем самый поздний факт
                - "highest_confidence": Берем с наибольшей уверенностью
                - "most_evidence": Берем с наибольшим количеством свидетельств
                - "manual": Требует ручного разрешения
        """
        self.strategy = strategy
        self.conflict_log = []
    
    def resolve(self, conflicting_facts: ConflictingFacts) -> Fact:
        """
        Разрешает конфликт между фактами
        
        Args:
            conflicting_facts: Конфликтующие факты
            
        Returns:
            Выбранный факт
        """
        resolved = conflicting_facts.resolve(self.strategy)
        
        # Логируем конфликт
        self.conflict_log.append({
            'timestamp': datetime.now(),
            'conflicting_values': conflicting_facts.get_all_values(),
            'resolved_value': resolved.object if resolved else None,
            'strategy': self.strategy,
            'reason': conflicting_facts.resolution_reason
        })
        
        return resolved
    
    def find_conflicts(self, facts: List[Fact]) -> List[ConflictingFacts]:
        """
        Находит все конфликты в списке фактов
        
        Args:
            facts: Список фактов
            
        Returns:
            Список групп конфликтующих фактов
        """
        conflicts = []
        processed = set()
        
        for i, fact1 in enumerate(facts):
            if fact1.id in processed:
                continue
            
            conflicting = ConflictingFacts([fact1])
            
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if fact2.id in processed:
                    continue
                
                if fact1.is_conflicting_with(fact2):
                    conflicting.add_fact(fact2)
                    processed.add(fact2.id)
            
            if len(conflicting.facts) > 1:
                conflicts.append(conflicting)
                processed.add(fact1.id)
        
        return conflicts
    
    def get_conflict_stats(self) -> Dict:
        """Возвращает статистику по конфликтам"""
        return {
            'total_conflicts': len(self.conflict_log),
            'strategy': self.strategy,
            'recent_conflicts': self.conflict_log[-10:]  # Последние 10 конфликтов
        }


class FactDatabase:
    """База данных для управления фактами"""
    
    def __init__(self, conflict_strategy: str = "latest"):
        """
        Инициализация базы данных фактов
        
        Args:
            conflict_strategy: Стратегия разрешения конфликтов
        """
        # Основное хранилище
        self.facts: Dict[str, Fact] = {}  # fact_id -> Fact
        self.dialogue_facts: Dict[str, List[str]] = defaultdict(list)  # dialogue_id -> [fact_ids]
        
        # Индекс для поиска
        self.index = FactIndex()
        
        # Разрешение конфликтов
        self.conflict_resolver = FactConflictResolver(conflict_strategy)
        
        # Статистика
        self.stats = FactStats()
        
        logger.info(f"База фактов инициализирована со стратегией конфликтов: {conflict_strategy}")
    
    def add_facts(self, dialogue_id: str, facts: List[Fact]):
        """
        Добавляет факты с дедупликацией и разрешением конфликтов
        
        Args:
            dialogue_id: ID диалога
            facts: Список фактов для добавления
        """
        if not facts:
            return
        
        # Получаем существующие факты диалога
        existing_ids = self.dialogue_facts.get(dialogue_id, [])
        existing_facts = [self.facts[fid] for fid in existing_ids if fid in self.facts]
        
        # Объединяем с новыми для поиска конфликтов
        all_facts = existing_facts + facts
        
        # Находим конфликты
        conflicts = self.conflict_resolver.find_conflicts(all_facts)
        
        # Разрешаем конфликты
        resolved_facts = {}
        for conflict_group in conflicts:
            resolved = self.conflict_resolver.resolve(conflict_group)
            if resolved:
                resolved_facts[resolved.id] = resolved
                self.stats.conflicts_resolved += 1
        
        # Добавляем факты
        for fact in facts:
            # Пропускаем если факт был частью конфликта и не был выбран
            if fact.id in resolved_facts:
                fact = resolved_facts[fact.id]
            elif any(fact in cg.facts and fact != cg.resolved_fact 
                    for cg in conflicts):
                continue
            
            # Проверяем дубликаты
            if fact.id not in self.facts:
                # Добавляем новый факт
                self.facts[fact.id] = fact
                self.dialogue_facts[dialogue_id].append(fact.id)
                self.index.add_fact(fact)
                
                # Обновляем статистику
                self._update_stats_on_add(fact)
            else:
                # Обновляем существующий факт (увеличиваем уверенность)
                existing = self.facts[fact.id]
                existing.confidence.update(fact.confidence.score)
        
        logger.debug(f"Добавлено {len(facts)} фактов для диалога {dialogue_id}")
    
    def get_facts(self, dialogue_id: str) -> List[Fact]:
        """
        Получает все факты диалога
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Список фактов
        """
        fact_ids = self.dialogue_facts.get(dialogue_id, [])
        return [self.facts[fid] for fid in fact_ids if fid in self.facts]
    
    def query_facts(self, dialogue_id: str, 
                   query: Optional[str] = None,
                   fact_type: Optional[FactType] = None,
                   min_confidence: float = 0.0) -> List[Fact]:
        """
        Поиск фактов по критериям
        
        Args:
            dialogue_id: ID диалога
            query: Текстовый запрос
            fact_type: Тип факта
            min_confidence: Минимальная уверенность
            
        Returns:
            Список найденных фактов
        """
        # Начинаем с фактов диалога
        dialogue_fact_ids = set(self.dialogue_facts.get(dialogue_id, []))
        
        if not dialogue_fact_ids:
            return []
        
        result_ids = dialogue_fact_ids
        
        # Фильтруем по типу
        if fact_type:
            type_ids = set(self.index.by_type[fact_type])
            result_ids = result_ids.intersection(type_ids)
        
        # Полнотекстовый поиск
        if query:
            text_ids = self.index.search_by_text(query)
            if text_ids:
                result_ids = result_ids.intersection(text_ids)
        
        # Получаем факты
        facts = []
        for fact_id in result_ids:
            if fact_id in self.facts:
                fact = self.facts[fact_id]
                if fact.confidence.score >= min_confidence:
                    facts.append(fact)
        
        # Сортируем по уверенности
        return sorted(facts, key=lambda f: f.confidence.score, reverse=True)
    
    def find_fact_by_type_and_subject(self, dialogue_id: str, 
                                     fact_type: FactType, 
                                     subject: str = "пользователь") -> Optional[Fact]:
        """
        Находит факт по типу и субъекту
        
        Args:
            dialogue_id: ID диалога
            fact_type: Тип факта
            subject: Субъект факта
            
        Returns:
            Найденный факт или None
        """
        type_facts = self.query_facts(dialogue_id, fact_type=fact_type)
        
        for fact in type_facts:
            if fact.subject == subject:
                return fact
        
        return None
    
    def get_user_profile(self, dialogue_id: str) -> Dict[str, Any]:
        """
        Создает профиль пользователя на основе фактов
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Структурированный профиль пользователя
        """
        facts = self.get_facts(dialogue_id)
        
        profile = {
            'personal': {},
            'work': {},
            'family': {},
            'education': {},
            'hobbies': [],
            'preferences': {},
            'health': {},
            'pets': [],
            'events': [],
            'contacts': {}
        }
        
        for fact in facts:
            # Личная информация
            if fact.type == FactType.PERSONAL_NAME:
                profile['personal']['name'] = fact.object
            elif fact.type == FactType.PERSONAL_AGE:
                profile['personal']['age'] = fact.object
            elif fact.type == FactType.PERSONAL_LOCATION:
                profile['personal']['location'] = fact.object
            elif fact.type == FactType.PERSONAL_GENDER:
                profile['personal']['gender'] = fact.object
            
            # Работа
            elif fact.type == FactType.WORK_OCCUPATION:
                profile['work']['occupation'] = fact.object
            elif fact.type == FactType.WORK_COMPANY:
                profile['work']['company'] = fact.object
            elif fact.type == FactType.WORK_POSITION:
                profile['work']['position'] = fact.object
            
            # Семья
            elif fact.type == FactType.FAMILY_SPOUSE:
                profile['family']['spouse'] = fact.object
            elif fact.type == FactType.FAMILY_CHILDREN:
                if 'children' not in profile['family']:
                    profile['family']['children'] = []
                profile['family']['children'].append(fact.object)
            
            # Образование
            elif fact.type == FactType.EDUCATION_INSTITUTION:
                profile['education']['institution'] = fact.object
            elif fact.type == FactType.EDUCATION_SPECIALITY:
                profile['education']['speciality'] = fact.object
            
            # Хобби
            elif fact.type in [FactType.HOBBY_SPORT, FactType.HOBBY_ACTIVITY]:
                profile['hobbies'].append(fact.object)
            
            # Предпочтения
            elif fact.type == FactType.PREFERENCE_FOOD:
                if 'food' not in profile['preferences']:
                    profile['preferences']['food'] = []
                profile['preferences']['food'].append(fact.object)
            
            # Здоровье
            elif fact.type == FactType.HEALTH_CONDITION:
                if 'conditions' not in profile['health']:
                    profile['health']['conditions'] = []
                profile['health']['conditions'].append(fact.object)
            
            # Питомцы
            elif fact.type in [FactType.PET_NAME, FactType.PET_TYPE, FactType.PET_BREED]:
                pet_info = {
                    'type': fact.object if fact.type == FactType.PET_TYPE else None,
                    'name': fact.object if fact.type == FactType.PET_NAME else None,
                    'breed': fact.object if fact.type == FactType.PET_BREED else None
                }
                profile['pets'].append(pet_info)
            
            # События
            elif fact.type in [FactType.EVENT_PAST, FactType.EVENT_FUTURE, FactType.EVENT_TRAVEL]:
                profile['events'].append({
                    'type': fact.type.value,
                    'description': fact.object
                })
        
        return profile
    
    def update_fact(self, fact_id: str, new_confidence: Optional[float] = None,
                   new_object: Optional[str] = None):
        """
        Обновляет существующий факт
        
        Args:
            fact_id: ID факта
            new_confidence: Новая уверенность
            new_object: Новое значение объекта
        """
        if fact_id not in self.facts:
            logger.warning(f"Факт {fact_id} не найден")
            return
        
        fact = self.facts[fact_id]
        
        if new_confidence is not None:
            fact.confidence.score = max(0.0, min(1.0, new_confidence))
        
        if new_object is not None:
            # Обновляем индексы
            self.index.remove_fact(fact)
            fact.object = new_object
            fact.id = fact.generate_id()  # Перегенерируем ID
            self.index.add_fact(fact)
    
    def delete_fact(self, fact_id: str):
        """
        Удаляет факт
        
        Args:
            fact_id: ID факта для удаления
        """
        if fact_id not in self.facts:
            return
        
        fact = self.facts[fact_id]
        
        # Удаляем из индексов
        self.index.remove_fact(fact)
        
        # Удаляем из диалога
        if fact.dialogue_id in self.dialogue_facts:
            self.dialogue_facts[fact.dialogue_id].remove(fact_id)
        
        # Удаляем сам факт
        del self.facts[fact_id]
        
        # Обновляем статистику
        self._update_stats_on_remove(fact)
    
    def clear_dialogue(self, dialogue_id: str):
        """
        Удаляет все факты диалога
        
        Args:
            dialogue_id: ID диалога
        """
        fact_ids = self.dialogue_facts.get(dialogue_id, []).copy()
        
        for fact_id in fact_ids:
            self.delete_fact(fact_id)
        
        if dialogue_id in self.dialogue_facts:
            del self.dialogue_facts[dialogue_id]
    
    def _update_stats_on_add(self, fact: Fact):
        """Обновляет статистику при добавлении факта"""
        self.stats.total_facts += 1
        
        # По типам
        type_name = fact.type.value
        self.stats.facts_by_type[type_name] = self.stats.facts_by_type.get(type_name, 0) + 1
        
        # По диалогам
        self.stats.facts_by_dialogue[fact.dialogue_id] = \
            self.stats.facts_by_dialogue.get(fact.dialogue_id, 0) + 1
        
        # Временные факты
        if isinstance(fact, TemporalFact):
            self.stats.temporal_facts += 1
        
        # Пересчитываем среднюю уверенность
        self._recalculate_average_confidence()
    
    def _update_stats_on_remove(self, fact: Fact):
        """Обновляет статистику при удалении факта"""
        self.stats.total_facts -= 1
        
        # По типам
        type_name = fact.type.value
        if type_name in self.stats.facts_by_type:
            self.stats.facts_by_type[type_name] -= 1
        
        # По диалогам
        if fact.dialogue_id in self.stats.facts_by_dialogue:
            self.stats.facts_by_dialogue[fact.dialogue_id] -= 1
        
        # Временные факты
        if isinstance(fact, TemporalFact):
            self.stats.temporal_facts -= 1
        
        # Пересчитываем среднюю уверенность
        self._recalculate_average_confidence()
    
    def _recalculate_average_confidence(self):
        """Пересчитывает среднюю уверенность"""
        if not self.facts:
            self.stats.average_confidence = 0.0
            return
        
        total_confidence = sum(f.confidence.score for f in self.facts.values())
        self.stats.average_confidence = total_confidence / len(self.facts)
    
    def get_stats(self) -> FactStats:
        """Возвращает статистику базы данных"""
        return self.stats
    
    def save(self, filepath: str):
        """
        Сохраняет базу данных на диск
        
        Args:
            filepath: Путь к файлу
        """
        data = {
            'facts': {fid: f.to_dict() for fid, f in self.facts.items()},
            'dialogue_facts': dict(self.dialogue_facts),
            'stats': self.stats.to_dict(),
            'conflict_log': self.conflict_resolver.conflict_log
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"База фактов сохранена: {self.stats.total_facts} фактов в {filepath}")
    
    def load(self, filepath: str):
        """
        Загружает базу данных с диска
        
        Args:
            filepath: Путь к файлу
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Очищаем текущую базу
        self.facts.clear()
        self.dialogue_facts.clear()
        self.index.clear()
        
        # Загружаем факты
        for fact_id, fact_dict in data['facts'].items():
            fact = Fact.from_dict(fact_dict)
            self.facts[fact_id] = fact
            self.index.add_fact(fact)
        
        # Загружаем связи с диалогами
        self.dialogue_facts = defaultdict(list, data['dialogue_facts'])
        
        # Загружаем статистику
        stats_dict = data.get('stats', {})
        self.stats = FactStats(**stats_dict)
        
        # Загружаем лог конфликтов
        self.conflict_resolver.conflict_log = data.get('conflict_log', [])
        
        logger.info(f"База фактов загружена: {self.stats.total_facts} фактов из {filepath}")



