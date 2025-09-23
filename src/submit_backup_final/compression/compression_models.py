"""
Модели данных для семантического сжатия
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class CompressionLevel(Enum):
    """Уровни сжатия информации"""
    NONE = "none"           # Без сжатия
    LIGHT = "light"         # Легкое сжатие (удаление стоп-слов)
    MODERATE = "moderate"   # Умеренное (извлечение ключевых фраз)
    AGGRESSIVE = "aggressive"  # Агрессивное (только факты)
    EXTREME = "extreme"     # Экстремальное (минимум информации)


class CompressionMethod(Enum):
    """Методы сжатия"""
    EXTRACTIVE = "extractive"    # Извлечение важных предложений
    ABSTRACTIVE = "abstractive"  # Генерация краткого пересказа
    TEMPLATE = "template"        # Заполнение шаблонов
    HYBRID = "hybrid"           # Комбинация методов


@dataclass
class CompressionConfig:
    """Конфигурация сжатия"""
    level: CompressionLevel = CompressionLevel.MODERATE
    method: CompressionMethod = CompressionMethod.HYBRID
    target_ratio: float = 0.3  # Целевой коэффициент сжатия
    min_length: int = 50       # Минимальная длина для сжатия
    max_length: int = 500      # Максимальная длина результата
    preserve_facts: bool = True  # Сохранять все факты
    preserve_names: bool = True  # Сохранять имена
    preserve_numbers: bool = True  # Сохранять числа
    use_cache: bool = True      # Кэшировать результаты
    language: str = "ru"        # Язык текста


@dataclass
class CompressionResult:
    """Результат сжатия"""
    original_text: str
    compressed_text: str
    compression_ratio: float
    method_used: CompressionMethod
    level_used: CompressionLevel
    facts_preserved: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def original_length(self) -> int:
        """Длина оригинального текста"""
        return len(self.original_text)
    
    @property
    def compressed_length(self) -> int:
        """Длина сжатого текста"""
        return len(self.compressed_text)
    
    @property
    def savings_percent(self) -> float:
        """Процент экономии места"""
        if self.original_length == 0:
            return 0.0
        return (1 - self.compressed_length / self.original_length) * 100
    
    def is_valid(self) -> bool:
        """Проверка валидности сжатия"""
        # Сжатый текст не должен быть пустым
        if not self.compressed_text.strip():
            return False
        
        # Сжатый текст должен быть короче оригинала
        if self.compressed_length >= self.original_length:
            return False
        
        # Должен достигать минимального коэффициента сжатия
        if self.compression_ratio > 0.9:  # Менее 10% сжатия
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'original_text': self.original_text[:200] + '...' if len(self.original_text) > 200 else self.original_text,
            'compressed_text': self.compressed_text,
            'compression_ratio': self.compression_ratio,
            'method': self.method_used.value,
            'level': self.level_used.value,
            'facts_preserved': self.facts_preserved,
            'keywords': self.keywords,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'savings_percent': self.savings_percent
        }


@dataclass
class CompressionStats:
    """Статистика сжатия"""
    total_processed: int = 0
    total_compressed: int = 0
    total_failed: int = 0
    total_original_chars: int = 0
    total_compressed_chars: int = 0
    average_ratio: float = 0.0
    best_ratio: float = 1.0
    worst_ratio: float = 0.0
    methods_used: Dict[str, int] = field(default_factory=dict)
    levels_used: Dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, result: CompressionResult, processing_time: float = 0):
        """Обновление статистики"""
        self.total_processed += 1
        
        if result.is_valid():
            self.total_compressed += 1
            self.total_original_chars += result.original_length
            self.total_compressed_chars += result.compressed_length
            
            # Обновляем средний коэффициент
            self.average_ratio = (
                self.total_compressed_chars / self.total_original_chars 
                if self.total_original_chars > 0 else 0
            )
            
            # Обновляем лучший/худший коэффициент
            self.best_ratio = min(self.best_ratio, result.compression_ratio)
            self.worst_ratio = max(self.worst_ratio, result.compression_ratio)
            
            # Счетчики методов и уровней
            method = result.method_used.value
            self.methods_used[method] = self.methods_used.get(method, 0) + 1
            
            level = result.level_used.value
            self.levels_used[level] = self.levels_used.get(level, 0) + 1
        else:
            self.total_failed += 1
        
        self.processing_time_ms += processing_time
    
    @property
    def success_rate(self) -> float:
        """Процент успешных сжатий"""
        if self.total_processed == 0:
            return 0.0
        return (self.total_compressed / self.total_processed) * 100
    
    @property
    def average_savings(self) -> float:
        """Средняя экономия места в процентах"""
        if self.average_ratio == 0:
            return 0.0
        return (1 - self.average_ratio) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Процент попаданий в кэш"""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return (self.cache_hits / total_cache_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'total_processed': self.total_processed,
            'total_compressed': self.total_compressed,
            'total_failed': self.total_failed,
            'success_rate': self.success_rate,
            'average_ratio': self.average_ratio,
            'average_savings': self.average_savings,
            'best_ratio': self.best_ratio,
            'worst_ratio': self.worst_ratio,
            'methods_used': self.methods_used,
            'levels_used': self.levels_used,
            'processing_time_ms': self.processing_time_ms,
            'cache_hit_rate': self.cache_hit_rate
        }
    
    def __str__(self) -> str:
        """Строковое представление"""
        return f"""CompressionStats:
  Processed: {self.total_processed}
  Success rate: {self.success_rate:.1f}%
  Average savings: {self.average_savings:.1f}%
  Best ratio: {self.best_ratio:.2f}
  Cache hit rate: {self.cache_hit_rate:.1f}%"""


@dataclass 
class TextSegment:
    """Сегмент текста для сжатия"""
    text: str
    importance: float = 1.0  # Важность сегмента (0-1)
    segment_type: str = "general"  # Тип: fact, opinion, description, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_preserve(self, threshold: float = 0.5) -> bool:
        """Определяет, нужно ли сохранить сегмент"""
        return self.importance >= threshold


@dataclass
class CompressionCache:
    """Кэш результатов сжатия"""
    cache: Dict[str, CompressionResult] = field(default_factory=dict)
    max_size: int = 1000
    hits: int = 0
    misses: int = 0
    
    def get(self, key: str) -> Optional[CompressionResult]:
        """Получение из кэша"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, result: CompressionResult):
        """Добавление в кэш"""
        # Очистка при переполнении (FIFO)
        if len(self.cache) >= self.max_size:
            # Удаляем первые 20% записей
            to_remove = list(self.cache.keys())[:int(self.max_size * 0.2)]
            for k in to_remove:
                del self.cache[k]
        
        self.cache[key] = result
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Процент попаданий"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0


@dataclass
class CompressionBatch:
    """Пакет текстов для батчевого сжатия"""
    texts: List[str]
    configs: List[CompressionConfig]
    session_ids: List[str]
    dialogue_id: str
    
    def __post_init__(self):
        """Валидация после создания"""
        # Проверяем, что списки одинаковой длины
        assert len(self.texts) == len(self.configs) == len(self.session_ids), \
            "Все списки должны быть одинаковой длины"
    
    @property
    def size(self) -> int:
        """Размер батча"""
        return len(self.texts)
    
    @property
    def total_chars(self) -> int:
        """Общее количество символов"""
        return sum(len(text) for text in self.texts)

