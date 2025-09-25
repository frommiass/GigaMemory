# modules/extraction/module.py

from typing import List, Dict, Any
from ...core.interfaces import IFactExtractor, ProcessingResult

from .fact_extractor import SmartFactExtractor
from .fact_database import FactDatabase
from .fact_models import Fact


class ExtractionModule(IFactExtractor):
    """Модуль извлечения фактов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Создаем извлекатель фактов
        # Здесь временно None вместо model_inference
        # В реальности будет передаваться через DI
        self.extractor = SmartFactExtractor(
            model_inference=None,
            use_rules_first=config.get('use_rules', True)
        )
        
        # База фактов
        self.database = FactDatabase(
            conflict_strategy=config.get('conflict_strategy', 'latest')
        )
    
    def extract_facts(self, text: str, context: Dict[str, Any]) -> ProcessingResult:
        """Извлекает факты из текста"""
        try:
            dialogue_id = context.get('dialogue_id', 'unknown')
            session_id = context.get('session_id', '0')
            
            # Извлекаем факты
            facts = self.extractor.extract_facts_from_text(text, session_id, dialogue_id)
            
            # Добавляем в базу
            if facts:
                self.database.add_facts(dialogue_id, facts)
            
            return ProcessingResult(
                success=True,
                data=facts,
                metadata={
                    'total_facts': len(facts),
                    'dialogue_id': dialogue_id,
                    'session_id': session_id
                }
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )
    
    def get_user_profile(self, dialogue_id: str) -> ProcessingResult:
        """Создает профиль пользователя на основе фактов"""
        try:
            profile = self.database.get_user_profile(dialogue_id)
            
            return ProcessingResult(
                success=True,
                data=profile,
                metadata={'dialogue_id': dialogue_id}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data={},
                metadata={},
                error=str(e)
            )
    
    def query_facts(self, dialogue_id: str, query: str) -> ProcessingResult:
        """Поиск фактов по запросу"""
        try:
            facts = self.database.query_facts(
                dialogue_id=dialogue_id,
                query=query,
                min_confidence=self.config.get('min_confidence', 0.5)
            )
            
            # Конвертируем факты в dict для сериализации
            facts_data = [fact.to_dict() for fact in facts]
            
            return ProcessingResult(
                success=True,
                data=facts_data,
                metadata={'found': len(facts)}
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                data=[],
                metadata={},
                error=str(e)
            )