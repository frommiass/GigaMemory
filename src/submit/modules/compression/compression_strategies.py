"""
Стратегии семантического сжатия текста
"""
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import Counter

from .compression_models import CompressionConfig, CompressionLevel, TextSegment


class CompressionStrategy(ABC):
    """Базовый класс для стратегий сжатия"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.stop_words = self._load_stop_words()
    
    @abstractmethod
    def compress(self, text: str) -> str:
        """Сжимает текст согласно стратегии"""
        pass
    
    def _load_stop_words(self) -> Set[str]:
        """Загружает стоп-слова для языка"""
        if self.config.language == "ru":
            return {
                'и', 'в', 'на', 'с', 'по', 'для', 'к', 'от', 'из', 'у', 'о',
                'а', 'но', 'да', 'или', 'если', 'что', 'как', 'это', 'то',
                'бы', 'же', 'ли', 'не', 'ни', 'очень', 'еще', 'уже', 'также',
                'только', 'можно', 'нужно', 'был', 'была', 'было', 'были',
                'есть', 'иметь', 'мочь', 'делать', 'сделать', 'так', 'такой',
                'весь', 'все', 'всё', 'вся', 'они', 'оно', 'она', 'он'
            }
        else:
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'about', 'as', 'is', 'was',
                'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does'
            }
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Извлекает ключевые слова из текста"""
        # Токенизация
        words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]+\b', text.lower())
        
        # Фильтрация стоп-слов
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Подсчет частотности
        word_freq = Counter(words)
        
        # TF-IDF упрощенный (только TF)
        total_words = len(words)
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
        
        # Возвращаем топ ключевые слова
        sorted_words = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлекает именованные сущности"""
        entities = {
            'names': [],
            'numbers': [],
            'dates': [],
            'locations': []
        }
        
        # Имена (слова с заглавной буквы)
        names = re.findall(r'\b[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)*\b', text)
        entities['names'] = list(set(names))
        
        # Числа
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
        entities['numbers'] = list(set(numbers))
        
        # Даты (простой паттерн)
        dates = re.findall(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', text)
        entities['dates'] = list(set(dates))
        
        # Локации (эвристика - слова после предлогов места)
        location_patterns = [
            r'(?:в|на|из|около|у|возле|рядом с)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)',
        ]
        for pattern in location_patterns:
            locations = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend(locations)
        entities['locations'] = list(set(entities['locations']))
        
        return entities
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        # Простое разбиение по знакам препинания
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def calculate_sentence_importance(self, sentence: str, keywords: List[str]) -> float:
        """Вычисляет важность предложения"""
        if not sentence:
            return 0.0
        
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Баллы за ключевые слова
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1.0
        
        # Бонус за именованные сущности
        entities = self.extract_entities(sentence)
        entity_count = sum(len(v) for v in entities.values())
        score += entity_count * 0.5
        
        # Штраф за очень короткие предложения
        if len(sentence.split()) < 3:
            score *= 0.5
        
        # Нормализация по длине
        word_count = len(sentence.split())
        if word_count > 0:
            score = score / word_count
        
        return min(1.0, score)


class ExtractiveStrategy(CompressionStrategy):
    """Экстрактивное сжатие - выбор важных предложений"""
    
    def compress(self, text: str) -> str:
        if len(text) < self.config.min_length:
            return text
        
        # Извлекаем ключевые слова
        keywords = self.extract_keywords(text, top_k=15)
        
        # Разбиваем на предложения
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return text
        
        # Вычисляем важность каждого предложения
        sentence_scores = []
        for sentence in sentences:
            importance = self.calculate_sentence_importance(sentence, keywords)
            sentence_scores.append((sentence, importance))
        
        # Сортируем по важности
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Определяем количество предложений для сохранения
        target_length = int(len(text) * self.config.target_ratio)
        
        selected_sentences = []
        current_length = 0
        
        for sentence, score in sentence_scores:
            if current_length >= target_length:
                break
            if score > 0.2 or len(selected_sentences) == 0:  # Минимальный порог важности
                selected_sentences.append(sentence)
                current_length += len(sentence)
        
        # Восстанавливаем исходный порядок
        result = []
        for sentence in sentences:
            if sentence in selected_sentences:
                result.append(sentence)
        
        compressed = '. '.join(result)
        if compressed and not compressed.endswith('.'):
            compressed += '.'
        
        return compressed


class AbstractiveStrategy(CompressionStrategy):
    """Абстрактивное сжатие - генерация краткого пересказа"""
    
    def __init__(self, config: CompressionConfig, model_inference=None):
        super().__init__(config)
        self.model = model_inference
    
    def compress(self, text: str) -> str:
        if len(text) < self.config.min_length:
            return text
        
        # Если нет модели, используем rule-based подход
        if not self.model:
            return self._rule_based_abstractive(text)
        
        # Используем LLM для генерации краткого пересказа
        return self._llm_abstractive(text)
    
    def _rule_based_abstractive(self, text: str) -> str:
        """Rule-based абстрактивное сжатие"""
        # Извлекаем ключевую информацию
        keywords = self.extract_keywords(text, top_k=10)
        entities = self.extract_entities(text)
        
        # Строим краткое резюме
        summary_parts = []
        
        # Добавляем имена
        if entities['names']:
            if len(entities['names']) == 1:
                summary_parts.append(f"Речь о {entities['names'][0]}")
            else:
                summary_parts.append(f"Упоминаются: {', '.join(entities['names'][:3])}")
        
        # Добавляем числа если есть
        if entities['numbers']:
            summary_parts.append(f"Числа: {', '.join(entities['numbers'][:3])}")
        
        # Добавляем локации
        if entities['locations']:
            summary_parts.append(f"Места: {', '.join(entities['locations'][:2])}")
        
        # Добавляем ключевые слова
        if keywords:
            summary_parts.append(f"Ключевые темы: {', '.join(keywords[:5])}")
        
        # Если ничего не нашли, берем первое предложение
        if not summary_parts:
            sentences = self.split_into_sentences(text)
            if sentences:
                return sentences[0]
        
        return '. '.join(summary_parts) + '.'
    
    def _llm_abstractive(self, text: str) -> str:
        """LLM-based абстрактивное сжатие"""
        from models import Message
        
        # Определяем целевую длину
        target_words = int(len(text.split()) * self.config.target_ratio)
        
        prompt = f"""Сожми текст до {target_words} слов, сохранив ключевую информацию.

Правила:
1. Сохрани все факты о людях (имена, возраст, профессии)
2. Сохрани все числа и даты
3. Используй телеграфный стиль
4. НЕ добавляй новую информацию

Текст: {text}

Сжатый текст:"""
        
        try:
            response = self.model.inference([Message('system', prompt)])
            compressed = response.strip()
            
            # Проверка валидности
            if len(compressed) > len(text) * 0.9:
                # Если сжатие не удалось, используем rule-based
                return self._rule_based_abstractive(text)
            
            return compressed
        except:
            # Fallback на rule-based
            return self._rule_based_abstractive(text)


class TemplateStrategy(CompressionStrategy):
    """Сжатие через заполнение шаблонов"""
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Загружает шаблоны для разных типов информации"""
        return {
            'person': "{name}, {age} лет, {occupation}",
            'work': "Работает {position} в {company}",
            'location': "Живет в {city}",
            'family': "Семья: {spouse}, дети: {children}",
            'hobby': "Увлечения: {hobbies}",
            'pet': "{pet_type} {pet_name}",
            'education': "Образование: {institution}, {speciality}",
            'contact': "Контакты: {phone}, {email}",
            'event': "{date}: {event_description}",
            'general': "{subject} {predicate} {object}"
        }
    
    def compress(self, text: str) -> str:
        if len(text) < self.config.min_length:
            return text
        
        # Извлекаем структурированную информацию
        extracted_data = self._extract_structured_data(text)
        
        # Заполняем подходящие шаблоны
        filled_templates = []
        
        for template_type, template in self.templates.items():
            if template_type in extracted_data:
                try:
                    filled = template.format(**extracted_data[template_type])
                    if filled and filled != template:  # Проверяем, что шаблон заполнен
                        filled_templates.append(filled)
                except KeyError:
                    continue
        
        # Если не удалось заполнить шаблоны, используем экстрактивный метод
        if not filled_templates:
            extractor = ExtractiveStrategy(self.config)
            return extractor.compress(text)
        
        # Объединяем заполненные шаблоны
        compressed = '. '.join(filled_templates)
        if compressed and not compressed.endswith('.'):
            compressed += '.'
        
        return compressed
    
    def _extract_structured_data(self, text: str) -> Dict[str, Dict[str, str]]:
        """Извлекает структурированные данные для заполнения шаблонов"""
        data = {}
        
        # Извлекаем информацию о человеке
        person_data = self._extract_person_info(text)
        if person_data:
            data['person'] = person_data
        
        # Извлекаем информацию о работе
        work_data = self._extract_work_info(text)
        if work_data:
            data['work'] = work_data
        
        # Извлекаем информацию о местоположении
        location_data = self._extract_location_info(text)
        if location_data:
            data['location'] = location_data
        
        # Извлекаем информацию о семье
        family_data = self._extract_family_info(text)
        if family_data:
            data['family'] = family_data
        
        return data
    
    def _extract_person_info(self, text: str) -> Optional[Dict[str, str]]:
        """Извлекает информацию о человеке"""
        info = {}
        
        # Имя
        name_pattern = r'(?:меня зовут|я\s*[-–]\s*)([А-ЯЁ][а-яё]+)'
        match = re.search(name_pattern, text, re.IGNORECASE)
        if match:
            info['name'] = match.group(1)
        
        # Возраст  
        age_pattern = r'(?:мне|возраст)\s*[-–]?\s*(\d+)\s*(?:лет|год)'
        match = re.search(age_pattern, text, re.IGNORECASE)
        if match:
            info['age'] = match.group(1)
        
        # Профессия
        occupation_pattern = r'(?:работаю|я)\s+([а-яё]+(?:ом|ером|истом))'
        match = re.search(occupation_pattern, text, re.IGNORECASE)
        if match:
            info['occupation'] = match.group(1)
        
        return info if info else None
    
    def _extract_work_info(self, text: str) -> Optional[Dict[str, str]]:
        """Извлекает информацию о работе"""
        info = {}
        
        # Должность
        position_pattern = r'(?:работаю|должность)\s+([а-яё]+)'
        match = re.search(position_pattern, text, re.IGNORECASE)
        if match:
            info['position'] = match.group(1)
        
        # Компания
        company_pattern = r'(?:в компании|в фирме|в)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)'
        match = re.search(company_pattern, text, re.IGNORECASE)
        if match:
            info['company'] = match.group(1)
        
        return info if info else None
    
    def _extract_location_info(self, text: str) -> Optional[Dict[str, str]]:
        """Извлекает информацию о местоположении"""
        info = {}
        
        # Город
        city_pattern = r'(?:живу в|проживаю в|из)\s+([А-ЯЁ][а-яё]+(?:\s*-?\s*[А-ЯЁ]?[а-яё]+)*)'
        match = re.search(city_pattern, text, re.IGNORECASE)
        if match:
            info['city'] = match.group(1)
        
        return info if info else None
    
    def _extract_family_info(self, text: str) -> Optional[Dict[str, str]]:
        """Извлекает информацию о семье"""
        info = {}
        
        # Супруг(а)
        spouse_pattern = r'(?:жена|муж|супруг|супруга)\s+([А-ЯЁ][а-яё]+)'
        match = re.search(spouse_pattern, text, re.IGNORECASE)
        if match:
            info['spouse'] = match.group(1)
        
        # Дети
        children_pattern = r'(\d+)\s+(?:детей|ребенок|ребенка)'
        match = re.search(children_pattern, text, re.IGNORECASE)
        if match:
            info['children'] = match.group(1)
        
        return info if info else None


def get_strategy(method: str, config: CompressionConfig, model=None) -> CompressionStrategy:
    """Фабричный метод для создания стратегии"""
    strategies = {
        'extractive': ExtractiveStrategy,
        'abstractive': lambda c: AbstractiveStrategy(c, model),
        'template': TemplateStrategy
    }
    
    strategy_class = strategies.get(method, ExtractiveStrategy)
    
    if callable(strategy_class):
        return strategy_class(config)
    else:
        return strategy_class
