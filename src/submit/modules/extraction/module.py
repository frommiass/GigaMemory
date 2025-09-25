# modules/extraction/module.py
"""
ExtractionModule - АВТОНОМНЫЙ модуль извлечения фактов для GigaMemory
Версия 3.1 - с улучшенным кэшированием и быстрым доступом к критическим фактам
"""

import sys
import os
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Добавляем путь к models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from models import Message
except ImportError:
    # Fallback если models недоступны
    Message = None
    logging.warning("Could not import Message model, using dict fallback")

from core.interfaces import IFactExtractor, ProcessingResult

logger = logging.getLogger(__name__)


class ExtractionModule(IFactExtractor):
    """
    АВТОНОМНЫЙ модуль извлечения фактов с улучшенным кэшированием
    """
    
    # TTL для разных типов фактов (в секундах)
    FACT_TTL = {
        'personal_name': 86400,        # 24 часа - имя редко меняется
        'personal_age': 43200,          # 12 часов - возраст меняется редко
        'personal_location': 3600,      # 1 час - может переехать
        'work_occupation': 7200,        # 2 часа
        'work_company': 7200,           # 2 часа
        'family_status': 21600,         # 6 часов
        'family_spouse': 86400,         # 24 часа
        'pet_name': 86400,              # 24 часа
        'default': 1800                 # 30 минут по умолчанию
    }
    
    # Критические факты для быстрого доступа
    CRITICAL_FACT_TYPES = [
        'personal_name',
        'personal_age',
        'personal_location',
        'work_occupation',
        'work_company',
        'family_status'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация автономного модуля
        
        Args:
            config: Конфигурация модуля
        """
        self.config = config
        
        # Импортируем ВНУТРЕННИЕ компоненты модуля
        from .fact_extractor import (
            RuleBasedFactExtractor, 
            SmartFactExtractor,
            HybridFactExtractor
        )
        from .fact_database import FactDatabase
        from .fact_models import (
            Fact, FactType, FactRelation, 
            FactConfidence, TemporalFact,
            ConflictingFacts
        )
        
        # Сохраняем классы для внутреннего использования
        self.Fact = Fact
        self.FactType = FactType
        self.FactRelation = FactRelation
        self.FactConfidence = FactConfidence
        self.TemporalFact = TemporalFact
        self.ConflictingFacts = ConflictingFacts
        
        # Выбираем стратегию извлечения
        use_llm = config.get('use_llm', False)
        use_rules = config.get('use_rules', True)
        
        if use_llm and use_rules:
            self.extractor = HybridFactExtractor(
                model_inference=None,
                rule_confidence_threshold=config.get('rule_confidence_threshold', 0.7)
            )
        elif use_rules:
            self.extractor = RuleBasedFactExtractor(
                min_confidence=config.get('min_confidence', 0.5)
            )
        else:
            self.extractor = SmartFactExtractor(
                model_inference=None,
                use_rules_first=False
            )
        
        # ВНУТРЕННЯЯ база данных фактов
        self.database = FactDatabase(
            conflict_strategy=config.get('conflict_strategy', 'latest')
        )
        
        # Кэш для критических фактов
        self.critical_facts_cache = {}
        self.cache_timestamps = {}
        
        # ЕДИНСТВЕННАЯ внешняя зависимость - optimizer для кэширования
        self.optimizer = None
        
        # Внутренняя статистика работы
        self.stats = {
            'total_extracted': 0,
            'facts_by_type': {},
            'conflicts_resolved': 0,
            'cache_hits': 0,
            'critical_cache_hits': 0,
            'copypaste_filtered': 0,
            'processing_time_ms': 0,
            'info_updates_detected': 0
        }
        
        logger.info(f"ExtractionModule v3.1 initialized: rules={use_rules}, llm={use_llm}")
    
    def set_dependencies(self, optimizer=None):
        """
        Устанавливает ТОЛЬКО optimizer для кэширования
        
        Args:
            optimizer: Модуль оптимизации для кэширования
        """
        self.optimizer = optimizer
        
        if self.optimizer and hasattr(self.optimizer, 'type_handler'):
            try:
                # Регистрируем обработчики для сериализации фактов
                self.optimizer.type_handler.register(
                    'fact',
                    compress_func=self._compress_fact,
                    decompress_func=self._decompress_fact
                )
                
                # Регистрируем обработчик для критических фактов
                self.optimizer.type_handler.register(
                    'critical_facts',
                    compress_func=lambda cf: cf,
                    decompress_func=lambda cf: cf
                )
            except Exception as e:
                logger.debug(f"Could not register handlers: {e}")
        
        logger.info(f"Dependencies set: optimizer={optimizer is not None}")
    
    def extract_facts(self, text: str, context: Dict[str, Any]) -> ProcessingResult:
        """
        АВТОНОМНО извлекает факты из текста с улучшенным кэшированием
        
        Args:
            text: Исходный текст для анализа
            context: Контекст с session_id и dialogue_id
            
        Returns:
            ProcessingResult с извлеченными фактами
        """
        start_time = time.time()
        
        try:
            session_id = context.get('session_id', '')
            dialogue_id = context.get('dialogue_id', '')
            
            # Валидация входных данных
            if not text:
                return ProcessingResult(
                    success=True,
                    data=[],
                    metadata={'empty_text': True}
                )
            
            # Обработка длинных текстов
            if len(text) > self.config.get('max_text_length', 10000):
                text = text[:self.config.get('max_text_length', 10000)]
                logger.warning(f"Text truncated to {len(text)} chars")
            
            # ФИЛЬТРАЦИЯ КОПИПАСТА
            if self.config.get('filter_copypaste', True) and self._is_copypaste(text):
                self.stats['copypaste_filtered'] += 1
                return ProcessingResult(
                    success=True,
                    data=[],
                    metadata={'filtered': 'copypaste', 'reason': 'no personal markers'}
                )
            
            # Определяем тип сообщения для оптимизации кэширования
            is_info_update = self._detect_info_update(text)
            if is_info_update:
                self.stats['info_updates_detected'] += 1
                # Для обновлений информации используем короткий TTL
                cache_ttl = 300  # 5 минут
            else:
                cache_ttl = 3600  # 1 час для обычных сообщений
            
            # КЭШИРОВАНИЕ с учетом типа сообщения
            cache_key = self._create_cache_key('extract', text[:200], dialogue_id, session_id)
            
            if self.optimizer and not is_info_update:  # Не используем кэш для обновлений
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True, 'cache_key': cache_key[:8]}
                    )
            
            # ИЗВЛЕЧЕНИЕ ФАКТОВ
            try:
                facts = self.extractor.extract_facts_from_text(
                    text=text,
                    session_id=session_id,
                    dialogue_id=dialogue_id
                )
                
                # Дополнительное извлечение для info_updating вопросов
                if is_info_update:
                    update_facts = self._extract_update_facts(text, session_id, dialogue_id)
                    facts.extend(update_facts)
                
            except Exception as extract_error:
                logger.warning(f"Extractor failed: {extract_error}")
                facts = []
            
            # Постобработка
            if facts:
                facts = self._postprocess_facts(facts, context)
            
            # Сохраняем во внутреннюю базу данных
            if facts and dialogue_id:
                self.database.add_facts(dialogue_id, facts)
                self.stats['conflicts_resolved'] += len(self.database.conflict_resolver.conflict_log)
                
                # Обновляем кэш критических фактов
                self._update_critical_facts_cache(dialogue_id, facts)
            
            # Обновляем статистику
            self.stats['total_extracted'] += len(facts)
            for fact in facts:
                fact_type_str = self._get_fact_type_string(fact)
                self.stats['facts_by_type'][fact_type_str] = \
                    self.stats['facts_by_type'].get(fact_type_str, 0) + 1
            
            # Кэшируем результат с TTL в зависимости от типа фактов
            if self.optimizer and facts:
                ttl = self._calculate_ttl_for_facts(facts)
                self.optimizer.cache_put(cache_key, facts, ttl=ttl)
            
            processing_time = (time.time() - start_time) * 1000
            self.stats['processing_time_ms'] = processing_time
            
            return ProcessingResult(
                success=True,
                data=facts,
                metadata={
                    'extracted_count': len(facts),
                    'session_id': session_id,
                    'dialogue_id': dialogue_id,
                    'processing_time_ms': processing_time,
                    'is_info_update': is_info_update,
                    'used_cache': False
                }
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Extraction failed: {str(e)}\n{error_details}")
            return ProcessingResult(
                success=False,
                data=[],
                error=f"Extraction failed: {str(e)}"
            )
    
    def get_critical_facts(self, dialogue_id: str) -> Dict[str, str]:
        """
        Возвращает критические факты для быстрого доступа
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            Словарь с критическими фактами
        """
        # Проверяем кэш критических фактов
        cache_key = f"critical_{dialogue_id}"
        
        # Проверяем свежесть кэша
        if cache_key in self.critical_facts_cache:
            cache_time = self.cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < 300:  # 5 минут
                self.stats['critical_cache_hits'] += 1
                return self.critical_facts_cache[cache_key]
        
        # Извлекаем критические факты из базы
        critical_facts = {}
        
        critical_types = {
            'name': self.FactType.PERSONAL_NAME,
            'age': self.FactType.PERSONAL_AGE,
            'location': self.FactType.PERSONAL_LOCATION,
            'occupation': self.FactType.WORK_OCCUPATION,
            'company': self.FactType.WORK_COMPANY,
            'marital_status': self.FactType.FAMILY_STATUS,
            'spouse': self.FactType.FAMILY_SPOUSE,
            'children_count': self.FactType.FAMILY_CHILDREN
        }
        
        for key, fact_type in critical_types.items():
            fact = self.database.find_fact_by_type_and_subject(
                dialogue_id, fact_type, 'пользователь'
            )
            if fact:
                critical_facts[key] = fact.object
                # Добавляем уверенность для критических фактов
                critical_facts[f"{key}_confidence"] = fact.confidence.score
        
        # Дополнительная информация о детях (количество)
        if 'children_count' in critical_facts:
            children_facts = self.database.query_facts(
                dialogue_id, 
                fact_type=self.FactType.FAMILY_CHILDREN
            )
            critical_facts['children_count'] = str(len(children_facts))
            critical_facts['children_names'] = [f.object for f in children_facts]
        
        # Кэшируем результат
        self.critical_facts_cache[cache_key] = critical_facts
        self.cache_timestamps[cache_key] = time.time()
        
        return critical_facts
    
    def get_user_profile(self, dialogue_id: str) -> ProcessingResult:
        """
        Строит профиль пользователя с критическими фактами
        """
        try:
            # Используем кэш для профилей
            cache_key = f"profile_{dialogue_id}"
            
            if self.optimizer:
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True}
                    )
            
            # Строим профиль из базы данных
            profile = self.database.get_user_profile(dialogue_id)
            
            # Добавляем критические факты для быстрого доступа
            critical_facts = self.get_critical_facts(dialogue_id)
            
            enhanced_profile = {
                'dialogue_id': dialogue_id,
                'critical_facts': critical_facts,  # Критические факты в начале
                'personal': profile.get('personal', {}),
                'family': profile.get('family', {}),
                'work': profile.get('work', {}),
                'pets': profile.get('pets', []),
                'education': profile.get('education', {}),
                'hobbies': profile.get('hobbies', []),
                'preferences': profile.get('preferences', {}),
                'health': profile.get('health', {}),
                'events': profile.get('events', []),
                'contacts': profile.get('contacts', {}),
                'stats': {
                    'total_facts': self.database.stats.total_facts,
                    'facts_by_type': self.database.stats.facts_by_type,
                    'conflicts_resolved': self.database.stats.conflicts_resolved,
                    'average_confidence': self.database.stats.average_confidence
                }
            }
            
            # Кэшируем профиль
            if self.optimizer:
                self.optimizer.cache_put(cache_key, enhanced_profile, ttl=300)
            
            return ProcessingResult(
                success=True,
                data=enhanced_profile,
                metadata={
                    'total_facts': self.database.stats.total_facts,
                    'has_critical_facts': bool(critical_facts),
                    'last_updated': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Profile creation failed: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=f"Profile creation failed: {str(e)}"
            )
    
    # === НОВЫЕ МЕТОДЫ ДЛЯ УЛУЧШЕННОГО ИЗВЛЕЧЕНИЯ ===
    
    def _detect_info_update(self, text: str) -> bool:
        """
        Определяет, является ли текст обновлением информации
        """
        update_markers = [
            'теперь', 'сейчас', 'изменилось', 'обновление',
            'больше не', 'уже не', 'стал', 'стала',
            'переехал', 'сменил', 'новый', 'новая',
            'был', 'была', 'раньше', 'прежде',
            'вышла замуж', 'женился', 'развелся',
            'родился', 'родилась', 'умер', 'умерла',
            'уволился', 'устроился', 'повысили'
        ]
        
        text_lower = text.lower()
        return any(marker in text_lower for marker in update_markers)
    
    def _extract_update_facts(self, text: str, session_id: str, dialogue_id: str) -> List:
        """
        Специальное извлечение для обновлений информации
        """
        import re
        update_facts = []
        
        # Паттерны для обнаружения изменений
        update_patterns = [
            # Изменение имени
            (r'теперь (?:меня зовут|мое имя)\s+([А-ЯЁ][а-яё]+)', 
             self.FactType.PERSONAL_NAME),
            
            # Изменение возраста
            (r'(?:мне уже|мне исполнилось|мне теперь)\s+(\d+)', 
             self.FactType.PERSONAL_AGE),
            
            # Изменение места жительства
            (r'переехал(?:а)?\s+в\s+([А-ЯЁ][а-яё]+)', 
             self.FactType.PERSONAL_LOCATION),
            
            # Изменение работы
            (r'(?:теперь работаю|устроился|устроилась)\s+(?:в\s+)?([А-ЯЁ][а-яё]+)',
             self.FactType.WORK_COMPANY),
            
            # Изменение семейного статуса
            (r'(?:женился|вышла замуж|теперь женат|теперь замужем)',
             self.FactType.FAMILY_STATUS),
            
            (r'(?:развелся|развелась|больше не женат|больше не замужем)',
             self.FactType.FAMILY_STATUS),
            
            # Новый питомец
            (r'(?:завел|завела|появился|появилась)\s+([а-яё]+)',
             self.FactType.PET_TYPE)
        ]
        
        for pattern_str, fact_type in update_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(text)
            
            if match:
                # Определяем значение для факта
                if fact_type == self.FactType.FAMILY_STATUS:
                    if 'женился' in text.lower() or 'женат' in text.lower():
                        value = 'женат'
                    elif 'замужем' in text.lower() or 'вышла замуж' in text.lower():
                        value = 'замужем'
                    elif 'развел' in text.lower():
                        value = 'разведен'
                    else:
                        continue
                else:
                    value = match.group(1) if match.groups() else match.group(0)
                
                # Создаем временной факт
                fact = self.TemporalFact(
                    type=fact_type,
                    subject='пользователь',
                    relation=self._get_relation_for_type(fact_type),
                    object=value,
                    confidence=self.FactConfidence(
                        score=0.95,  # Высокая уверенность для явных обновлений
                        source='info_update'
                    ),
                    session_id=session_id,
                    dialogue_id=dialogue_id,
                    raw_text=text[:200],
                    is_current=True,  # Это текущая информация
                    timestamp=datetime.now()
                )
                
                update_facts.append(fact)
        
        return update_facts
    
    def _calculate_ttl_for_facts(self, facts: List) -> int:
        """
        Рассчитывает TTL для кэширования на основе типов фактов
        """
        if not facts:
            return self.FACT_TTL['default']
        
        # Берем минимальный TTL среди всех фактов
        min_ttl = self.FACT_TTL['default']
        
        for fact in facts:
            fact_type_str = self._get_fact_type_string(fact)
            ttl = self.FACT_TTL.get(fact_type_str, self.FACT_TTL['default'])
            min_ttl = min(min_ttl, ttl)
        
        return min_ttl
    
    def _update_critical_facts_cache(self, dialogue_id: str, facts: List):
        """
        Обновляет кэш критических фактов при добавлении новых
        """
        cache_key = f"critical_{dialogue_id}"
        
        # Если кэш существует, обновляем только изменившиеся факты
        if cache_key in self.critical_facts_cache:
            critical_facts = self.critical_facts_cache[cache_key]
            
            for fact in facts:
                fact_type_str = self._get_fact_type_string(fact)
                
                # Обновляем только критические факты
                if fact_type_str in self.CRITICAL_FACT_TYPES:
                    # Маппинг типов фактов на ключи в critical_facts
                    key_map = {
                        'personal_name': 'name',
                        'personal_age': 'age',
                        'personal_location': 'location',
                        'work_occupation': 'occupation',
                        'work_company': 'company',
                        'family_status': 'marital_status'
                    }
                    
                    key = key_map.get(fact_type_str)
                    if key:
                        critical_facts[key] = fact.object
                        critical_facts[f"{key}_confidence"] = fact.confidence.score
                        
            # Обновляем временную метку
            self.cache_timestamps[cache_key] = time.time()
    
    def _compress_fact(self, fact) -> Dict:
        """
        Сжимает факт для кэширования
        """
        return {
            't': self._get_fact_type_string(fact),  # type
            'o': fact.object,                        # object
            'c': round(fact.confidence.score, 2),    # confidence
            's': fact.subject[:10] if len(fact.subject) > 10 else fact.subject,  # subject
            'r': fact.relation.value if hasattr(fact.relation, 'value') else str(fact.relation)[:10]
        }
    
    def _decompress_fact(self, data: Dict):
        """
        Восстанавливает факт из кэша
        """
        try:
            fact_type = self.FactType(data['t'])
        except:
            # Fallback для неизвестных типов
            fact_type = self.FactType.GENERAL
        
        return self.Fact(
            type=fact_type,
            subject=data.get('s', 'пользователь'),
            relation=data.get('r', 'is'),
            object=data['o'],
            confidence=self.FactConfidence(
                score=data['c'],
                source='cached'
            ),
            session_id='',
            dialogue_id=''
        )
    
    def _get_fact_type_string(self, fact) -> str:
        """
        Безопасно получает строковое представление типа факта
        """
        if hasattr(fact.type, 'value'):
            return fact.type.value
        return str(fact.type)
    
    def _get_relation_for_type(self, fact_type) -> str:
        """
        Определяет отношение для типа факта
        """
        from .fact_patterns import get_relation_for_type
        relation = get_relation_for_type(fact_type)
        
        if hasattr(relation, 'value'):
            return relation.value
        return str(relation)
    
    def _is_copypaste(self, text: str) -> bool:
        """
        Определяет является ли текст копипастом
        """
        # Признаки копипаста
        if len(text) > 5000:
            return True
        
        # Проверяем личные маркеры
        personal_markers = ['я ', 'меня', 'мой', 'моя', 'мое', 'мне', 'у меня']
        personal_count = sum(text.lower().count(marker) for marker in personal_markers)
        
        # Если мало личных маркеров - вероятно копипаст
        if len(text) > 1000 and personal_count < 3:
            return True
        
        # Проверяем специфические признаки
        copypaste_indicators = [
            text.count('\n\n') > 10,
            'википедия' in text.lower(),
            'copyright' in text.lower(),
            '©' in text,
            text.count('http') > 3,
            'источник:' in text.lower(),
        ]
        
        return sum(copypaste_indicators) >= 2
    
    def _postprocess_facts(self, facts: List, context: Dict) -> List:
        """
        Постобработка для улучшения качества фактов
        """
        processed = []
        
        for fact in facts:
            # Повышаем уверенность для критических фактов
            fact_type_str = self._get_fact_type_string(fact)
            if fact_type_str in ['personal_name', 'personal_age']:
                fact.confidence.score = min(1.0, fact.confidence.score * 1.2)
            
            # Нормализуем значения
            if fact.type == self.FactType.PERSONAL_AGE:
                try:
                    age = int(''.join(filter(str.isdigit, fact.object)))
                    if 0 < age < 150:
                        fact.object = str(age)
                    else:
                        continue
                except:
                    continue
            
            processed.append(fact)
        
        return processed
    
    def _create_cache_key(self, *args) -> str:
        """
        Создает ключ для кэша
        """
        key_str = '_'.join(str(arg)[:50] for arg in args if arg)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # === МЕТОДЫ СОВМЕСТИМОСТИ С ПРЕДЫДУЩЕЙ ВЕРСИЕЙ ===
    
    def query_facts(self, dialogue_id: str, query: str) -> ProcessingResult:
        """Поиск фактов (для совместимости)"""
        try:
            # Проверяем критические факты сначала
            if any(word in query.lower() for word in ['имя', 'зовут', 'возраст', 'работа']):
                critical = self.get_critical_facts(dialogue_id)
                if critical:
                    # Преобразуем в список фактов
                    facts = []
                    for key, value in critical.items():
                        if not key.endswith('_confidence') and value:
                            # Создаем псевдо-факт для ответа
                            fact = type('Fact', (), {
                                'object': value,
                                'subject': 'пользователь',
                                'relation': 'is',
                                'confidence': type('Confidence', (), {
                                    'score': critical.get(f'{key}_confidence', 0.8)
                                })(),
                                'to_natural_text': lambda: f"пользователь - {value}"
                            })()
                            facts.append(fact)
                    
                    if facts:
                        return ProcessingResult(
                            success=True,
                            data=facts[:5],
                            metadata={'from_critical_cache': True}
                        )
            
            # Fallback к обычному поиску
            facts = self.database.query_facts(
                dialogue_id=dialogue_id,
                query=query,
                min_confidence=self.config.get('min_confidence', 0.5)
            )
            
            return ProcessingResult(
                success=True,
                data=facts[:10],
                metadata={'query': query, 'found_count': len(facts)}
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ProcessingResult(
                success=False,
                data=[],
                error=str(e)
            )
    
    def get_stats(self) -> Dict:
        """Возвращает статистику работы модуля"""
        return {
            'module_stats': self.stats,
            'database_stats': self.database.get_stats().to_dict() if hasattr(self.database, 'get_stats') else {},
            'extractor_stats': self.extractor.get_stats() if hasattr(self.extractor, 'get_stats') else {},
            'cache_info': {
                'critical_facts_cached': len(self.critical_facts_cache),
                'cache_timestamps': len(self.cache_timestamps)
            }
        }
    
    def clear_dialogue(self, dialogue_id: str):
        """Очищает все данные диалога включая кэши"""
        # Очищаем базу данных
        self.database.clear_dialogue(dialogue_id)
        
        # Очищаем кэши критических фактов
        cache_key = f"critical_{dialogue_id}"
        if cache_key in self.critical_facts_cache:
            del self.critical_facts_cache[cache_key]
        if cache_key in self.cache_timestamps:
            del self.cache_timestamps[cache_key]
        
        logger.info(f"Dialogue {dialogue_id} fully cleared including caches")