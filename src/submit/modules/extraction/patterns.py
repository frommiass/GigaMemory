# src/submit/modules/extraction/patterns.py
"""
Упрощенные паттерны для извлечения фактов
"""
import re
from typing import Dict, List, Pattern, Optional
from dataclasses import dataclass


@dataclass
class FactPattern:
    """Паттерн для извлечения факта"""
    type: str
    patterns: List[Pattern]
    confidence: float = 0.7
    

class FactPatterns:
    """Класс с паттернами для извлечения фактов"""
    
    def __init__(self):
        """Инициализация паттернов"""
        self.patterns = self._create_patterns()
    
    def _create_patterns(self) -> Dict[str, FactPattern]:
        """Создает базовые паттерны для извлечения"""
        return {
            'name': FactPattern(
                type='personal_name',
                patterns=[
                    re.compile(r'меня зовут\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'я\s+[-–—]\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'мое имя\s+[-–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                ],
                confidence=0.9
            ),
            
            'age': FactPattern(
                type='personal_age',
                patterns=[
                    re.compile(r'мне\s+(\d+)\s*(?:лет|год)', re.IGNORECASE),
                    re.compile(r'(\d+)\s*(?:лет|год)(?:а|ов)?', re.IGNORECASE),
                    re.compile(r'возраст\s*[:–—]\s*(\d+)', re.IGNORECASE),
                ],
                confidence=0.85
            ),
            
            'location': FactPattern(
                type='personal_location',
                patterns=[
                    re.compile(r'живу\s+в\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'из\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'город\s*[:–—]\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                ],
                confidence=0.8
            ),
            
            'work': FactPattern(
                type='work_occupation',
                patterns=[
                    re.compile(r'работаю\s+([а-яё]+(?:ом|ером|ором))', re.IGNORECASE),
                    re.compile(r'я\s+([а-яё]+(?:ист|ер|ор))', re.IGNORECASE),
                    re.compile(r'профессия\s*[:–—]\s*([а-яё]+)', re.IGNORECASE),
                ],
                confidence=0.75
            ),
            
            'company': FactPattern(
                type='work_company',
                patterns=[
                    re.compile(r'работаю\s+в\s+([^,.]+)', re.IGNORECASE),
                    re.compile(r'компания\s*[:–—]\s*([^,.]+)', re.IGNORECASE),
                ],
                confidence=0.7
            ),
            
            'hobby': FactPattern(
                type='hobby',
                patterns=[
                    re.compile(r'люблю\s+([а-яё]+)', re.IGNORECASE),
                    re.compile(r'увлекаюсь\s+([а-яё]+)', re.IGNORECASE),
                    re.compile(r'хобби\s*[:–—]\s*([а-яё]+)', re.IGNORECASE),
                ],
                confidence=0.65
            ),
            
            'pet': FactPattern(
                type='pet',
                patterns=[
                    re.compile(r'(?:кот|кошка|пес|собака)\s+(?:по имени\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'питомец\s*[:–—]\s*([а-яё]+)', re.IGNORECASE),
                ],
                confidence=0.7
            ),
            
            'family': FactPattern(
                type='family',
                patterns=[
                    re.compile(r'(?:жена|муж|супруг)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'(?:сын|дочь)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
                    re.compile(r'(\d+)\s+(?:детей|ребенок)', re.IGNORECASE),
                ],
                confidence=0.75
            ),
        }
    
    def extract_from_text(self, text: str) -> List[Dict[str, any]]:
        """
        Извлекает факты из текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Список извлеченных фактов
        """
        facts = []
        
        for pattern_name, fact_pattern in self.patterns.items():
            for pattern in fact_pattern.patterns:
                match = pattern.search(text)
                if match:
                    fact = {
                        'type': fact_pattern.type,
                        'value': match.group(1),
                        'confidence': fact_pattern.confidence,
                        'pattern': pattern_name,
                        'raw_text': text[:100]
                    }
                    facts.append(fact)
                    break  # Берем первое совпадение для каждого типа
        
        return facts
    
    def get_pattern_by_type(self, fact_type: str) -> Optional[FactPattern]:
        """Получает паттерн по типу факта"""
        for pattern in self.patterns.values():
            if pattern.type == fact_type:
                return pattern
        return None