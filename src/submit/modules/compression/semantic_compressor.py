"""
Семантический компрессор - основной класс для сжатия текста
"""
import time
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

from .compression_models import (
    CompressionConfig, CompressionResult, CompressionStats, 
    CompressionLevel, CompressionMethod, CompressionCache
)
from .compression_strategies import (
    CompressionStrategy, ExtractiveStrategy, AbstractiveStrategy, 
    TemplateStrategy, get_strategy
)

logger = logging.getLogger(__name__)


class SemanticCompressor:
    """Основной класс для семантического сжатия текста"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Инициализация компрессора
        
        Args:
            config: Конфигурация сжатия
        """
        self.config = config or CompressionConfig()
        self.stats = CompressionStats()
        self.cache = CompressionCache() if self.config.use_cache else None
        
        # Инициализируем стратегии
        self.strategies = self._init_strategies()
        
        logger.info(f"SemanticCompressor инициализирован с конфигом: {self.config.level.value}")
    
    def _init_strategies(self) -> Dict[str, CompressionStrategy]:
        """Инициализирует доступные стратегии"""
        strategies = {}
        
        # Экстрактивная стратегия (всегда доступна)
        strategies['extractive'] = ExtractiveStrategy(self.config)
        
        # Абстрактивная стратегия (требует модель)
        if hasattr(self.config, 'llm_model') and self.config.llm_model:
            strategies['abstractive'] = AbstractiveStrategy(self.config, self.config.llm_model)
        
        # Шаблонная стратегия
        strategies['template'] = TemplateStrategy(self.config)
        
        return strategies
    
    def compress(self, text: str, 
                level: Optional[CompressionLevel] = None,
                method: Optional[CompressionMethod] = None) -> CompressionResult:
        """
        Сжимает текст
        
        Args:
            text: Исходный текст
            level: Уровень сжатия
            method: Метод сжатия
            
        Returns:
            Результат сжатия
        """
        start_time = time.time()
        
        # Используем параметры из конфига если не указаны
        level = level or self.config.level
        method = method or self.config.method
        
        # Проверяем кэш
        cache_key = self._get_cache_key(text, level, method)
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.stats.cache_hits += 1
                logger.debug(f"Результат найден в кэше для ключа: {cache_key}")
                return cached_result
            self.stats.cache_misses += 1
        
        try:
            # Выбираем стратегию
            strategy = self._select_strategy(method)
            
            # Сжимаем текст
            compressed_text = strategy.compress(text)
            
            # Создаем результат
            result = CompressionResult(
                original_text=text,
                compressed_text=compressed_text,
                compression_ratio=len(compressed_text) / len(text) if len(text) > 0 else 1.0,
                method_used=method,
                level_used=level,
                facts_preserved=self._extract_preserved_facts(text, compressed_text),
                keywords=self._extract_keywords(compressed_text),
                metadata={
                    'strategy_name': strategy.__class__.__name__,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            )
            
            # Обновляем статистику
            processing_time = (time.time() - start_time) * 1000
            self.stats.update(result, processing_time)
            
            # Сохраняем в кэш
            if self.cache:
                self.cache.put(cache_key, result)
            
            logger.debug(f"Текст сжат: {len(text)} -> {len(compressed_text)} символов")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка сжатия текста: {e}")
            self.stats.total_failed += 1
            
            # Возвращаем исходный текст в случае ошибки
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                compression_ratio=1.0,
                method_used=method,
                level_used=level,
                metadata={'error': str(e)}
            )
    
    def compress_batch(self, texts: List[str], 
                      levels: Optional[List[CompressionLevel]] = None,
                      methods: Optional[List[CompressionMethod]] = None) -> List[CompressionResult]:
        """
        Батчевое сжатие текстов
        
        Args:
            texts: Список текстов
            levels: Список уровней сжатия
            methods: Список методов сжатия
            
        Returns:
            Список результатов сжатия
        """
        results = []
        
        # Нормализуем параметры
        levels = levels or [self.config.level] * len(texts)
        methods = methods or [self.config.method] * len(texts)
        
        for text, level, method in zip(texts, levels, methods):
            result = self.compress(text, level, method)
            results.append(result)
        
        return results
    
    def _select_strategy(self, method: CompressionMethod) -> CompressionStrategy:
        """Выбирает стратегию для метода"""
        method_name = method.value
        
        if method_name in self.strategies:
            return self.strategies[method_name]
        
        # Fallback на экстрактивную стратегию
        logger.warning(f"Стратегия {method_name} недоступна, используем extractive")
        return self.strategies['extractive']
    
    def _get_cache_key(self, text: str, level: CompressionLevel, method: CompressionMethod) -> str:
        """Создает ключ для кэша"""
        import hashlib
        
        # Создаем хэш от текста и параметров
        content = f"{text[:100]}_{level.value}_{method.value}_{self.config.target_ratio}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_preserved_facts(self, original: str, compressed: str) -> List[str]:
        """Извлекает сохраненные факты"""
        # Простая эвристика - ищем числа, имена, даты
        import re
        
        facts = []
        
        # Числа
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', compressed)
        facts.extend([f"число: {num}" for num in numbers])
        
        # Имена (слова с заглавной буквы)
        names = re.findall(r'\b[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)*\b', compressed)
        facts.extend([f"имя: {name}" for name in names])
        
        # Даты
        dates = re.findall(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', compressed)
        facts.extend([f"дата: {date}" for date in dates])
        
        return facts[:10]  # Максимум 10 фактов
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые слова из сжатого текста"""
        if not text:
            return []
        
        # Используем экстрактивную стратегию для извлечения ключевых слов
        extractor = ExtractiveStrategy(self.config)
        return extractor.extract_keywords(text, top_k=5)
    
    def get_stats(self) -> CompressionStats:
        """Возвращает статистику работы компрессора"""
        return self.stats
    
    def clear_cache(self):
        """Очищает кэш"""
        if self.cache:
            self.cache.clear()
            logger.info("Кэш очищен")
    
    def update_config(self, new_config: CompressionConfig):
        """Обновляет конфигурацию компрессора"""
        self.config = new_config
        self.strategies = self._init_strategies()
        logger.info("Конфигурация обновлена")


class RuleBasedCompressor(SemanticCompressor):
    """Компрессор на основе правил"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        super().__init__(config)
        # Ограничиваем только правилами
        self.strategies = {'extractive': ExtractiveStrategy(self.config)}
        logger.info("RuleBasedCompressor инициализирован")


class LLMCompressor(SemanticCompressor):
    """Компрессор с использованием LLM"""
    
    def __init__(self, config: CompressionConfig, model_inference):
        """
        Инициализация LLM компрессора
        
        Args:
            config: Конфигурация
            model_inference: Модель для инференса
        """
        super().__init__(config)
        
        # Добавляем LLM стратегию
        self.strategies['abstractive'] = AbstractiveStrategy(self.config, model_inference)
        
        # Устанавливаем абстрактивный метод по умолчанию
        self.config.method = CompressionMethod.ABSTRACTIVE
        
        logger.info("LLMCompressor инициализирован")


class HybridCompressor(SemanticCompressor):
    """Гибридный компрессор, комбинирующий разные методы"""
    
    def __init__(self, config: Optional[CompressionConfig] = None, model_inference=None):
        super().__init__(config)
        
        # Добавляем LLM стратегию если доступна
        if model_inference:
            self.strategies['abstractive'] = AbstractiveStrategy(self.config, model_inference)
        
        # Устанавливаем гибридный метод по умолчанию
        self.config.method = CompressionMethod.HYBRID
        
        logger.info("HybridCompressor инициализирован")
    
    def compress(self, text: str, 
                level: Optional[CompressionLevel] = None,
                method: Optional[CompressionMethod] = None) -> CompressionResult:
        """Гибридное сжатие с выбором лучшего результата"""
        
        # Если метод не указан или HYBRID, пробуем несколько стратегий
        if not method or method == CompressionMethod.HYBRID:
            results = []
            
            # Пробуем экстрактивную стратегию
            try:
                extractive_result = self.strategies['extractive'].compress(text)
                results.append(('extractive', extractive_result))
            except Exception as e:
                logger.warning(f"Ошибка экстрактивного сжатия: {e}")
            
            # Пробуем абстрактивную стратегию если доступна
            if 'abstractive' in self.strategies:
                try:
                    abstractive_result = self.strategies['abstractive'].compress(text)
                    results.append(('abstractive', abstractive_result))
                except Exception as e:
                    logger.warning(f"Ошибка абстрактивного сжатия: {e}")
            
            # Пробуем шаблонную стратегию
            try:
                template_result = self.strategies['template'].compress(text)
                results.append(('template', template_result))
            except Exception as e:
                logger.warning(f"Ошибка шаблонного сжатия: {e}")
            
            # Выбираем лучший результат
            if results:
                best_strategy, best_result = self._select_best_result(results, level or self.config.level)
                
                # Создаем финальный результат
                final_result = CompressionResult(
                    original_text=text,
                    compressed_text=best_result,
                    compression_ratio=len(best_result) / len(text) if len(text) > 0 else 1.0,
                    method_used=CompressionMethod.HYBRID,
                    level_used=level or self.config.level,
                    facts_preserved=self._extract_preserved_facts(text, best_result),
                    keywords=self._extract_keywords(best_result),
                    metadata={
                        'best_strategy': best_strategy,
                        'all_strategies_tried': [r[0] for r in results]
                    }
                )
                
                return final_result
        
        # Fallback на обычное сжатие
        return super().compress(text, level, method)
    
    def _select_best_result(self, results: List[tuple], level: CompressionLevel) -> tuple:
        """Выбирает лучший результат из списка"""
        if not results:
            raise ValueError("Нет результатов для выбора")
        
        # Простая эвристика выбора
        target_ratios = {
            CompressionLevel.LIGHT: 0.8,
            CompressionLevel.MODERATE: 0.5,
            CompressionLevel.AGGRESSIVE: 0.3,
            CompressionLevel.EXTREME: 0.1
        }
        
        target_ratio = target_ratios.get(level, 0.5)
        
        # Выбираем результат, наиболее близкий к целевому коэффициенту
        best_strategy, best_result = min(
            results,
            key=lambda x: abs(len(x[1]) / len(x[1]) - target_ratio) if len(x[1]) > 0 else float('inf')
        )
        
        return best_strategy, best_result
