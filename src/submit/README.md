# 🧠 GigaMemory - Интеллектуальная система памяти

Полнофункциональная система памяти с векторным поиском, извлечением фактов и семантическим сжатием.

## 📁 Структура проекта

```
src/submit/
├── smart_memory_optimized.py    # 🧠 Главный оптимизированный класс
├── config.yaml                  # ⚙️ Конфигурация системы
├── config_loader.py             # 📋 Загрузчик конфигурации
│
├── optimization/                # ⚡ Оптимизация
│   ├── cache_manager.py        # Умное кэширование
│   └── batch_processor.py      # Батчевая обработка
│
├── embeddings/                  # 🔍 Векторный поиск
│   ├── embedding_engine.py     # Создание эмбеддингов
│   └── vector_store.py         # Векторное хранилище
│
├── extraction/                  # 📝 Извлечение фактов
│   ├── fact_extractor.py       # Извлекатель фактов
│   ├── fact_database.py        # База фактов
│   └── fact_patterns.py        # Паттерны извлечения
│
├── compression/                 # 🗜️ Сжатие
│   ├── semantic_compressor.py  # Семантическое сжатие
│   └── hierarchical_compressor.py # Иерархическое сжатие
│
└── rag/                        # 🎯 RAG система
    ├── vector_rag_engine.py    # Векторный RAG
    └── compressed_rag_engine.py # RAG со сжатием
```

## 🚀 Основные компоненты

### 1. **OptimizedSmartMemory** - Главный класс
- Полная интеграция всех компонентов
- Кэширование и батчевая обработка
- Автоматическое сохранение состояния
- Мониторинг и метрики

### 2. **Система конфигурации**
- `config.yaml` - YAML конфигурация
- `config_loader.py` - Загрузчик с автопоиском
- Поддержка переменных окружения

### 3. **Оптимизация производительности**
- **CacheManager** - Умное кэширование с TTL и стратегиями вытеснения
- **BatchProcessor** - Батчевая обработка с приоритетами
- **EmbeddingBatchProcessor** - Специализированный для эмбеддингов
- **FactExtractionBatchProcessor** - Специализированный для фактов

### 4. **Векторный поиск**
- **EmbeddingEngine** - Создание эмбеддингов
- **VectorStore** - Эффективное хранение и поиск
- Поддержка различных метрик (cosine, euclidean)

### 5. **Извлечение фактов**
- **SmartFactExtractor** - Гибридный извлекатель (LLM + правила)
- **FactDatabase** - База фактов с разрешением конфликтов
- **FactPatterns** - Паттерны для извлечения

### 6. **Семантическое сжатие**
- **SemanticCompressor** - Многоуровневое сжатие
- **HierarchicalCompressor** - Иерархическое сжатие
- Стратегии: extractive, abstractive, template, hybrid

### 7. **RAG система**
- **VectorRAGEngine** - Базовый векторный RAG
- **CompressedRAGEngine** - RAG с интегрированным сжатием
- Гибридный поиск (векторный + keyword)

## ⚙️ Конфигурация

### Основные настройки в `config.yaml`:

```yaml
# Модель
model:
  path: "/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16"
  device: "cuda"

# Векторный поиск
embedding:
  model_name: "cointegrated/rubert-tiny2"
  batch_size: 32

# Сжатие
compression:
  enabled: true
  level: "moderate"
  method: "hybrid"

# Кэширование
cache:
  max_size: 10000
  max_memory_mb: 1024
  eviction_strategy: "lru"
```

## 💡 Примеры использования

### Базовое использование:

```python
from submit.smart_memory_optimized import OptimizedSmartMemory

# Создание системы
memory = OptimizedSmartMemory()

# Обработка диалога
stats = memory.process_dialogue_optimized("dialogue_1", messages)
print(f"Обработано: {stats['sessions_count']} сессий")

# Ответ на вопрос
answer = memory.answer_question_optimized("dialogue_1", "Расскажи о работе")
print(answer)
```

### С кастомной конфигурацией:

```python
from submit.config_loader import config_manager

# Загрузка конфигурации
config = config_manager.load_config("custom_config.yaml")

# Создание с конфигурацией
memory = OptimizedSmartMemory(config.model_path)
```

### Мониторинг системы:

```python
# Получение статистики
stats = memory.get_system_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
print(f"Compression ratio: {stats['metrics']['compression_ratio']:.2f}")
```

## 🔧 Оптимизации

### 1. **Кэширование**
- Многоуровневое кэширование (embeddings, facts, compression, queries)
- TTL и стратегии вытеснения (LRU, LFU, FIFO)
- Автоматическое управление памятью

### 2. **Батчевая обработка**
- Параллельная обработка эмбеддингов
- Приоритетные очереди задач
- Адаптивные размеры батчей

### 3. **Фоновая обработка**
- Асинхронная обработка батчей
- Автосохранение состояния
- Мониторинг производительности

### 4. **Персистентность**
- Сохранение векторов, фактов и кэша
- Автоматические бэкапы
- Восстановление состояния

## 📊 Метрики и мониторинг

Система отслеживает:
- **Производительность**: время ответа, throughput
- **Кэш**: hit rate, размер, вытеснения
- **Сжатие**: коэффициенты сжатия, экономия места
- **Факты**: количество извлеченных фактов, уверенность
- **Очереди**: размеры батчей, время ожидания

## 🎯 Преимущества

1. **Высокая производительность** - кэширование и батчевая обработка
2. **Масштабируемость** - иерархическое сжатие и векторный поиск
3. **Точность** - гибридный поиск и извлечение фактов
4. **Надежность** - автосохранение и восстановление состояния
5. **Гибкость** - конфигурируемые компоненты и стратегии

## 🚀 Готовность к продакшену

- ✅ Полная интеграция всех компонентов
- ✅ Конфигурируемость через YAML
- ✅ Мониторинг и метрики
- ✅ Автосохранение и восстановление
- ✅ Обработка ошибок и логирование
- ✅ Потокобезопасность
- ✅ Оптимизация производительности

Система готова к использованию в продакшене! 🎉
