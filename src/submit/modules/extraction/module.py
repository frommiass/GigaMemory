# modules/extraction/module.py
"""
ExtractionModule - АВТОНОМНЫЙ модуль извлечения фактов для GigaMemory
Версия 3.0 - полностью изолированный, взаимодействует только через Orchestrator
"""

from core.interfaces import IFactExtractor, ProcessingResult
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class ExtractionModule(IFactExtractor):
    """
    АВТОНОМНЫЙ модуль извлечения фактов
    Не знает о других модулях кроме Optimizer для кэширования
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация автономного модуля
        
        Args:
            config: Конфигурация модуля
                - min_confidence: Минимальная уверенность для фактов (default: 0.5)
                - use_llm: Использовать LLM для извлечения (default: False)
                - use_rules: Использовать правила (default: True)
                - conflict_strategy: Стратегия разрешения конфликтов (default: 'latest')
                - filter_copypaste: Фильтровать копипаст (default: True)
                - max_text_length: Максимальная длина текста (default: 10000)
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
            # Гибридный подход
            self.extractor = HybridFactExtractor(
                model_inference=None,  # LLM пока не используем
                rule_confidence_threshold=config.get('rule_confidence_threshold', 0.7)
            )
        elif use_rules:
            # Только правила - БЫСТРО и НАДЕЖНО
            self.extractor = RuleBasedFactExtractor(
                min_confidence=config.get('min_confidence', 0.5)
            )
        else:
            # Только LLM
            self.extractor = SmartFactExtractor(
                model_inference=None,
                use_rules_first=False
            )
        
        # ВНУТРЕННЯЯ база данных фактов с версионированием
        self.database = FactDatabase(
            conflict_strategy=config.get('conflict_strategy', 'latest')
        )
        
        # ЕДИНСТВЕННАЯ внешняя зависимость - optimizer для кэширования
        self.optimizer = None
        
        # Внутренняя статистика работы
        self.stats = {
            'total_extracted': 0,
            'facts_by_type': {},
            'conflicts_resolved': 0,
            'cache_hits': 0,
            'copypaste_filtered': 0,
            'processing_time_ms': 0
        }
        
        logger.info(f"ExtractionModule initialized (AUTONOMOUS): rules={use_rules}, llm={use_llm}")
    
    def set_dependencies(self, optimizer=None):
        """
        Устанавливает ТОЛЬКО optimizer для кэширования
        Модуль остается полностью автономным
        
        Args:
            optimizer: Модуль оптимизации для кэширования (опционально)
        """
        self.optimizer = optimizer
        
        # Регистрируем обработчики для кэширования фактов
        if self.optimizer and hasattr(self.optimizer, 'type_handler'):
            try:
                self.optimizer.type_handler.register(
                    'fact',
                    compress_func=lambda f: {
                        'type': f.type.value,
                        'object': f.object,
                        'confidence': f.confidence.score
                    },
                    decompress_func=lambda d: self._create_fact_from_cache(d)
                )
            except Exception as e:
                logger.debug(f"Could not register fact handler: {e}")
        
        logger.info(f"Dependencies set: optimizer={optimizer is not None}")
    
    def extract_facts(self, text: str, context: Dict[str, Any]) -> ProcessingResult:
        """
        АВТОНОМНО извлекает факты из текста
        
        Args:
            text: Исходный текст для анализа
            context: Контекст с session_id и dialogue_id
            
        Returns:
            ProcessingResult с извлеченными фактами
        """
        import time
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
            
            if len(text) > self.config.get('max_text_length', 10000):
                # Обрезаем текст вместо отказа
                text = text[:self.config.get('max_text_length', 10000)]
                logger.warning(f"Text truncated to {len(text)} chars")
            
            # ФИЛЬТРАЦИЯ КОПИПАСТА - критично для конкурса!
            if self.config.get('filter_copypaste', True) and self._is_copypaste(text):
                self.stats['copypaste_filtered'] += 1
                return ProcessingResult(
                    success=True,
                    data=[],
                    metadata={'filtered': 'copypaste', 'reason': 'no personal markers'}
                )
            
            # КЭШИРОВАНИЕ - единственное внешнее взаимодействие
            cache_key = self._create_cache_key('extract', text[:200], dialogue_id, session_id)
            
            if self.optimizer:
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True, 'cache_key': cache_key[:8]}
                    )
            
            # АВТОНОМНОЕ ИЗВЛЕЧЕНИЕ ФАКТОВ
            facts = self.extractor.extract_facts_from_text(
                text=text,
                session_id=session_id,
                dialogue_id=dialogue_id
            )
            
            # Постобработка - улучшение качества фактов
            if facts:
                facts = self._postprocess_facts(facts, context)
            
            # Сохраняем во ВНУТРЕННЮЮ базу данных
            if facts and dialogue_id:
                self.database.add_facts(dialogue_id, facts)
                self.stats['conflicts_resolved'] += self.database.conflict_resolver.conflict_log.__len__()
            
            # Обновляем внутреннюю статистику
            self.stats['total_extracted'] += len(facts)
            for fact in facts:
                fact_type = fact.type.value
                self.stats['facts_by_type'][fact_type] = \
                    self.stats['facts_by_type'].get(fact_type, 0) + 1
            
            # Кэшируем результат
            if self.optimizer and facts:
                self.optimizer.cache_put(cache_key, facts, ttl=3600)
            
            # Измеряем производительность
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
                    'used_cache': False
                }
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ProcessingResult(
                success=False,
                data=[],
                error=f"Extraction failed: {str(e)}"
            )
    
    def get_user_profile(self, dialogue_id: str) -> ProcessingResult:
        """
        Строит профиль пользователя из ВНУТРЕННЕЙ базы фактов
        
        Args:
            dialogue_id: ID диалога
            
        Returns:
            ProcessingResult с полным профилем пользователя
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
            
            # Строим профиль из ВНУТРЕННЕЙ базы данных
            profile = self.database.get_user_profile(dialogue_id)
            
            # Дополняем критическими фактами для конкурса
            enhanced_profile = {
                'dialogue_id': dialogue_id,
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
                
                # Критические факты для быстрого доступа
                'critical_facts': {
                    'name': self._get_critical_fact(dialogue_id, self.FactType.PERSONAL_NAME),
                    'age': self._get_critical_fact(dialogue_id, self.FactType.PERSONAL_AGE),
                    'location': self._get_critical_fact(dialogue_id, self.FactType.PERSONAL_LOCATION),
                    'marital_status': self._get_critical_fact(dialogue_id, self.FactType.FAMILY_STATUS),
                    'occupation': self._get_critical_fact(dialogue_id, self.FactType.WORK_OCCUPATION)
                },
                
                # Статистика
                'stats': {
                    'total_facts': self.database.stats.total_facts,
                    'facts_by_type': self.database.stats.facts_by_type,
                    'conflicts_resolved': self.database.stats.conflicts_resolved,
                    'average_confidence': self.database.stats.average_confidence
                }
            }
            
            # Кэшируем профиль на 5 минут
            if self.optimizer:
                self.optimizer.cache_put(cache_key, enhanced_profile, ttl=300)
            
            return ProcessingResult(
                success=True,
                data=enhanced_profile,
                metadata={
                    'total_facts': self.database.stats.total_facts,
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
    
    def query_facts(self, dialogue_id: str, query: str) -> ProcessingResult:
        """
        АВТОНОМНЫЙ поиск фактов во ВНУТРЕННЕЙ базе данных
        
        Args:
            dialogue_id: ID диалога
            query: Поисковый запрос
            
        Returns:
            ProcessingResult с найденными фактами
        """
        try:
            # Кэшируем частые запросы
            cache_key = self._create_cache_key('query', dialogue_id, query)
            
            if self.optimizer:
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True, 'query': query}
                    )
            
            # Определяем тип факта из запроса
            fact_type = self._detect_fact_type_from_query(query)
            
            # АВТОНОМНЫЙ поиск во ВНУТРЕННЕЙ базе данных
            facts = self.database.query_facts(
                dialogue_id=dialogue_id,
                query=query,
                fact_type=fact_type,
                min_confidence=self.config.get('min_confidence', 0.5)
            )
            
            # Дополнительный поиск по ключевым словам
            if not facts and query:
                facts = self._fallback_keyword_search(dialogue_id, query)
            
            # Сортируем по релевантности
            facts = sorted(facts, key=lambda f: f.confidence.score, reverse=True)
            
            # Ограничиваем количество
            facts = facts[:10]
            
            # Кэшируем результат на 15 минут
            if self.optimizer and facts:
                self.optimizer.cache_put(cache_key, facts, ttl=900)
            
            return ProcessingResult(
                success=True,
                data=facts,
                metadata={
                    'query': query,
                    'found_count': len(facts),
                    'fact_type': fact_type.value if fact_type else None,
                    'used_fallback': len(facts) > 0 and not fact_type
                }
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ProcessingResult(
                success=False,
                data=[],
                error=f"Query failed: {str(e)}"
            )
    
    # === ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ДЛЯ КОНКУРСА ===
    
    def get_fact_timeline(self, dialogue_id: str, fact_type: str) -> List[Dict]:
        """
        История изменения факта во времени
        
        Args:
            dialogue_id: ID диалога
            fact_type: Тип факта
            
        Returns:
            Timeline изменений факта
        """
        try:
            # Преобразуем строку в FactType
            fact_type_enum = self._parse_fact_type(fact_type)
            if not fact_type_enum:
                return []
            
            # Получаем все версии факта из ВНУТРЕННЕЙ базы
            facts = self.database.query_facts(
                dialogue_id=dialogue_id,
                fact_type=fact_type_enum
            )
            
            # Строим timeline
            timeline = []
            for fact in facts:
                entry = {
                    'value': fact.object,
                    'confidence': fact.confidence.score,
                    'session_id': fact.session_id,
                    'timestamp': fact.extracted_at.isoformat() if hasattr(fact, 'extracted_at') else None,
                    'is_current': True
                }
                
                if isinstance(fact, self.TemporalFact):
                    entry['is_current'] = fact.is_current
                
                timeline.append(entry)
            
            # Сортируем по времени
            timeline.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '')
            
            # Помечаем только последний как текущий
            if timeline:
                for entry in timeline[:-1]:
                    entry['is_current'] = False
            
            return timeline
            
        except Exception as e:
            logger.error(f"Timeline generation failed: {e}")
            return []
    
    def resolve_fact_conflicts(self, facts: List[Any]) -> Any:
        """
        Разрешает конфликты между версиями фактов
        
        Args:
            facts: Список конфликтующих фактов
            
        Returns:
            Выбранный факт после разрешения
        """
        if not facts:
            return None
        
        if len(facts) == 1:
            return facts[0]
        
        # Используем ВНУТРЕННИЙ resolver
        conflicting = self.ConflictingFacts(facts=facts)
        resolved = self.database.conflict_resolver.resolve(conflicting)
        
        self.stats['conflicts_resolved'] += 1
        
        return resolved
    
    def get_facts_for_prompt(self, dialogue_id: str, question: str) -> str:
        """
        Форматирует топ-5 фактов для промпта
        ИСПОЛЬЗУЕТСЯ В ORCHESTRATOR ДЛЯ FALLBACK
        
        Args:
            dialogue_id: ID диалога
            question: Вопрос пользователя
            
        Returns:
            Отформатированная строка с фактами
        """
        result = self.query_facts(dialogue_id, question)
        
        if not result.success or not result.data:
            return ""
        
        facts_lines = []
        for i, fact in enumerate(result.data[:5], 1):
            if hasattr(fact, 'to_natural_text'):
                fact_text = fact.to_natural_text()
            else:
                fact_text = f"{fact.subject} {fact.relation} {fact.object}"
            
            confidence = fact.confidence.score if hasattr(fact, 'confidence') else 0.5
            facts_lines.append(f"{i}. {fact_text} (уверенность: {confidence:.0%})")
        
        if facts_lines:
            return "Известные факты о пользователе:\n" + "\n".join(facts_lines)
        return ""
    
    # === ВНУТРЕННИЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===
    
    def _is_copypaste(self, text: str) -> bool:
        """Определяет является ли текст копипастом"""
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
        """Постобработка для улучшения качества фактов"""
        processed = []
        
        for fact in facts:
            # Повышаем уверенность для критических фактов
            if fact.type in [self.FactType.PERSONAL_NAME, self.FactType.PERSONAL_AGE]:
                fact.confidence.score = min(1.0, fact.confidence.score * 1.2)
            
            # Нормализуем значения
            if fact.type == self.FactType.PERSONAL_AGE:
                # Убедимся что возраст - число
                try:
                    age = int(''.join(filter(str.isdigit, fact.object)))
                    if 0 < age < 150:
                        fact.object = str(age)
                    else:
                        continue  # Пропускаем невалидный возраст
                except:
                    continue
            
            processed.append(fact)
        
        return processed
    
    def _detect_fact_type_from_query(self, query: str) -> Optional:
        """Определяет тип факта из запроса"""
        query_lower = query.lower()
        
        # Маппинг ключевых слов на типы фактов
        type_keywords = {
            self.FactType.PERSONAL_NAME: ['имя', 'зовут', 'фио', 'как меня', 'как зовут'],
            self.FactType.PERSONAL_AGE: ['возраст', 'сколько лет', 'лет'],
            self.FactType.PERSONAL_LOCATION: ['где живу', 'город', 'адрес', 'где я живу'],
            self.FactType.WORK_OCCUPATION: ['профессия', 'работа', 'кем работаю', 'должность'],
            self.FactType.WORK_COMPANY: ['где работаю', 'компания', 'фирма'],
            self.FactType.FAMILY_STATUS: ['женат', 'замужем', 'холост', 'семейное положение'],
            self.FactType.FAMILY_SPOUSE: ['жена', 'муж', 'супруг'],
            self.FactType.FAMILY_CHILDREN: ['дети', 'сын', 'дочь', 'ребенок'],
            self.FactType.PET_NAME: ['питомец', 'кот', 'кошка', 'собака', 'кличка'],
            self.FactType.PET_TYPE: ['какие питомцы', 'животные'],
            self.FactType.HOBBY_ACTIVITY: ['хобби', 'увлечения', 'интересы'],
        }
        
        for fact_type, keywords in type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return fact_type
        
        return None
    
    def _fallback_keyword_search(self, dialogue_id: str, query: str) -> List:
        """Резервный поиск по ключевым словам"""
        all_facts = self.database.get_facts(dialogue_id)
        query_words = set(query.lower().split())
        
        relevant_facts = []
        for fact in all_facts:
            fact_text = f"{fact.subject} {fact.object}".lower()
            fact_words = set(fact_text.split())
            
            # Считаем пересечение слов
            overlap = len(query_words & fact_words)
            if overlap > 0:
                # Временно увеличиваем score для сортировки
                fact._temp_relevance = overlap / len(query_words)
                relevant_facts.append(fact)
        
        # Сортируем по релевантности
        relevant_facts.sort(key=lambda f: f._temp_relevance, reverse=True)
        
        # Убираем временное поле
        for fact in relevant_facts:
            delattr(fact, '_temp_relevance')
        
        return relevant_facts[:5]
    
    def _parse_fact_type(self, type_str: str):
        """Парсит строку в FactType"""
        try:
            return self.FactType(type_str)
        except ValueError:
            # Пробуем найти по частичному совпадению
            type_str_upper = type_str.upper()
            for fact_type in self.FactType:
                if type_str_upper in fact_type.value.upper():
                    return fact_type
            return None
    
    def _get_critical_fact(self, dialogue_id: str, fact_type) -> Optional[str]:
        """Получает критический факт для конкурса"""
        fact = self.database.find_fact_by_type_and_subject(
            dialogue_id, fact_type, 'пользователь'
        )
        return fact.object if fact else None
    
    def _create_fact_from_cache(self, data: Dict):
        """Создает факт из кэшированных данных"""
        return self.Fact(
            type=self.FactType(data['type']),
            subject='пользователь',
            relation=self.FactRelation.IS,
            object=data['object'],
            confidence=self.FactConfidence(
                score=data['confidence'],
                source='cached'
            ),
            session_id='',
            dialogue_id=''
        )
    
    def _create_cache_key(self, *args) -> str:
        """Создает ключ для кэша"""
        key_str = '_'.join(str(arg)[:50] for arg in args if arg)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # === МЕТОДЫ ДЛЯ ОТЛАДКИ И МОНИТОРИНГА ===
    
    def get_stats(self) -> Dict:
        """Возвращает статистику работы модуля"""
        return {
            'module_stats': self.stats,
            'database_stats': self.database.get_stats().to_dict() if hasattr(self.database, 'get_stats') else {},
            'extractor_stats': self.extractor.get_stats() if hasattr(self.extractor, 'get_stats') else {}
        }
    
    def clear_dialogue(self, dialogue_id: str):
        """Очищает все факты диалога"""
        self.database.clear_dialogue(dialogue_id)
        logger.info(f"Dialogue {dialogue_id} cleared from internal database")