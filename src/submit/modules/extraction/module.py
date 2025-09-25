# modules/extraction/module.py
from core.interfaces import IFactExtractor, ProcessingResult
from typing import Dict, Any, List, Optional

class ExtractionModule(IFactExtractor):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        from .fact_extractor import SmartFactExtractor
        from .fact_database import FactDatabase
        from .patterns import FactPatterns
        
        self.extractor = SmartFactExtractor(
            min_confidence=config.get('min_confidence', 0.6),
            use_llm=config.get('use_llm', False),
            use_rules=config.get('use_rules', True)
        )
        
        self.database = FactDatabase(
            conflict_strategy=config.get('conflict_strategy', 'latest')
        )
        
        self.patterns = FactPatterns()
        self.optimizer = None
        
    def set_optimizer(self, optimizer):
        """Устанавливает оптимизатор для кэширования"""
        self.optimizer = optimizer
    
    def extract_facts(self, text: str, context: Dict[str, Any]) -> ProcessingResult:
        """Извлекает факты из текста"""
        try:
            # Кэшируем результаты извлечения
            if self.optimizer:
                # Ключ на основе первых 100 символов и контекста
                cache_key = self._create_cache_key(
                    'facts',
                    text[:100],
                    context.get('dialogue_id'),
                    context.get('session_id')
                )
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True}
                    )
            
            # Извлекаем факты
            facts = self.extractor.extract_from_text(
                text=text,
                session_id=context.get('session_id'),
                dialogue_id=context.get('dialogue_id')
            )
            
            # Сохраняем в базу если есть dialogue_id
            if facts and context.get('dialogue_id'):
                self.database.add_facts(context['dialogue_id'], facts)
            
            # Кэшируем результат
            if self.optimizer and facts:
                self.optimizer.cache_put(cache_key, facts, ttl=3600)
            
            return ProcessingResult(
                success=True,
                data=facts,
                metadata={
                    'extracted_count': len(facts),
                    'context': context
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                error=f"Extraction failed: {str(e)}"
            )
    
    def get_user_profile(self, dialogue_id: str) -> ProcessingResult:
        """Создает профиль пользователя на основе фактов"""
        try:
            # Получаем все факты
            facts = self.database.get_all_facts(dialogue_id)
            
            if not facts:
                return ProcessingResult(
                    success=True,
                    data={},
                    metadata={'dialogue_id': dialogue_id, 'empty': True}
                )
            
            # Группируем по типам
            profile = {
                'personal': [],
                'preferences': [],
                'work': [],
                'schedule': [],
                'other': []
            }
            
            for fact in facts:
                category = self._categorize_fact(fact)
                profile[category].append({
                    'content': fact.content,
                    'confidence': fact.confidence,
                    'timestamp': fact.timestamp
                })
            
            # Удаляем пустые категории
            profile = {k: v for k, v in profile.items() if v}
            
            return ProcessingResult(
                success=True,
                data=profile,
                metadata={
                    'dialogue_id': dialogue_id,
                    'total_facts': len(facts)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                error=f"Profile creation failed: {str(e)}"
            )
    
    def query_facts(self, dialogue_id: str, query: str) -> ProcessingResult:
        """Поиск фактов по запросу"""
        try:
            # Используем кэш для частых запросов
            if self.optimizer:
                cache_key = self._create_cache_key('query', dialogue_id, query)
                cached = self.optimizer.cache_get(cache_key)
                if cached:
                    return ProcessingResult(
                        success=True,
                        data=cached,
                        metadata={'from_cache': True}
                    )
            
            # Ищем релевантные факты
            facts = self.database.query_facts(
                dialogue_id=dialogue_id,
                query=query,
                min_confidence=self.config.get('min_confidence', 0.5)
            )
            
            # Кэшируем на 15 минут
            if self.optimizer and facts:
                self.optimizer.cache_put(cache_key, facts, ttl=900)
            
            return ProcessingResult(
                success=True,
                data=facts,
                metadata={
                    'query': query,
                    'found_count': len(facts)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                error=f"Query failed: {str(e)}"
            )
    
    # Дополнительные методы для интеграции с RAG
    def get_facts_for_prompt(self, dialogue_id: str, question: str) -> str:
        """Возвращает отформатированные факты для промпта"""
        result = self.query_facts(dialogue_id, question)
        
        if not result.success or not result.data:
            return ""
        
        facts_lines = []
        for fact in result.data[:5]:  # Топ-5 фактов
            if hasattr(fact, 'to_natural_text'):
                facts_lines.append(f"• {fact.to_natural_text()}")
            else:
                facts_lines.append(f"• {fact.get('content', str(fact))}")
        
        return "\n".join(facts_lines)
    
    def export_profile_markdown(self, dialogue_id: str) -> str:
        """Экспорт профиля в Markdown"""
        profile_result = self.get_user_profile(dialogue_id)
        
        if not profile_result.success:
            return f"# Профиль недоступен\nОшибка: {profile_result.error}"
        
        profile = profile_result.data
        md_lines = [f"# Профиль пользователя\n**Dialogue ID**: {dialogue_id}\n"]
        
        for category, facts in profile.items():
            md_lines.append(f"\n## {category.title()}")
            for fact in facts:
                confidence = fact.get('confidence', 0)
                md_lines.append(f"- {fact['content']} *(уверенность: {confidence:.0%})*")
        
        return "\n".join(md_lines)
    
    def _categorize_fact(self, fact) -> str:
        """Категоризирует факт"""
        content = fact.content.lower() if hasattr(fact, 'content') else str(fact).lower()
        
        if any(w in content for w in ['зовут', 'имя', 'возраст', 'живу', 'родился']):
            return 'personal'
        elif any(w in content for w in ['люблю', 'нравится', 'предпочитаю', 'хобби']):
            return 'preferences'
        elif any(w in content for w in ['работаю', 'профессия', 'должность', 'компания']):
            return 'work'
        elif any(w in content for w in ['встреча', 'планы', 'завтра', 'сегодня']):
            return 'schedule'
        else:
            return 'other'
    
    def _create_cache_key(self, *args) -> str:
        """Создает ключ для кэша"""
        import hashlib
        key_str = '_'.join(str(arg)[:50] for arg in args if arg)
        return hashlib.md5(key_str.encode()).hexdigest()