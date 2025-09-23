# 🔍 Анализ использования filters/ и storage/

## 📊 ОБЩАЯ СТАТИСТИКА

### filters/ - АКТИВНО ИСПОЛЬЗУЕТСЯ
- **Файлов:** 6 файлов
- **Используется в:** 5 основных файлах системы
- **Статус:** ✅ Полностью активен

### storage/ - АКТИВНО ИСПОЛЬЗУЕТСЯ  
- **Файлов:** 5 файлов
- **Используется в:** 3 основных файлах системы
- **Статус:** ✅ Полностью активен

## 🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ИСПОЛЬЗОВАНИЯ

### 📁 filters/ - ГДЕ ИСПОЛЬЗУЕТСЯ

#### 1. `session_grouper.py` - АКТИВНО ИСПОЛЬЗУЕТСЯ
**Импортируется в:**
- `model_inference_optimized.py` - строка 75: `from .filters.session_grouper import SessionGrouper`
- `smart_memory.py` - строка 116: `from .filters.session_grouper import SessionGrouper`
- `smart_memory.py` - строка 125: `from .filters.session_grouper import extract_session_content`
- `smart_memory_optimized.py` - строка 157: `from .filters.session_grouper import SessionGrouper`
- `smart_memory_optimized.py` - строка 167: `from .filters.session_grouper import extract_session_content`

**Функции:**
- `SessionGrouper` - группировка сообщений по сессиям
- `extract_session_content` - извлечение содержимого сессий

#### 2. `message_cleaner.py` - АКТИВНО ИСПОЛЬЗУЕТСЯ
**Импортируется в:**
- `model_inference_optimized.py` - строка 96: `from .filters.message_cleaner import is_copy_paste_content, is_personal_message`

**Функции:**
- `is_copy_paste_content` - определение копипаста
- `is_personal_message` - определение личных сообщений
- `clean_messages` - очистка сообщений
- `get_message_quality_score` - оценка качества сообщений

#### 3. `relevance_filter.py` - ИСПОЛЬЗУЕТСЯ В RAG
**Импортируется в:**
- `rag/compressed_rag_engine.py` - для фильтрации релевантности
- `rag/engine.py` - для базового RAG

**Функции:**
- `RelevanceFilter` - фильтр релевантности
- `find_relevant_sessions` - поиск релевантных сессий

#### 4. `keyword_matcher.py` - ИСПОЛЬЗУЕТСЯ В RAG
**Импортируется в:**
- `rag/compressed_rag_engine.py` - для поиска по ключевым словам
- `rag/interface.py` - для интерфейса RAG

**Функции:**
- `KeywordMatcher` - поиск по ключевым словам
- `find_messages_by_topic` - поиск сообщений по теме

#### 5. `regex_patterns.py` - ИСПОЛЬЗУЕТСЯ ВНУТРИ filters/
**Импортируется в:**
- `message_cleaner.py` - для паттернов очистки
- `session_grouper.py` - для паттернов группировки

**Содержит:**
- `COMPILED_PATTERNS` - скомпилированные регулярные выражения
- `PERSONAL_MARKERS` - маркеры личных сообщений
- `COPYPASTE_MARKERS` - маркеры копипаста
- `TECH_SIGNS` - технические признаки

#### 6. `__init__.py` - ЭКСПОРТИРУЕТ ВСЕ ФУНКЦИИ
**Экспортирует все функции из всех модулей filters/**

### 📁 storage/ - ГДЕ ИСПОЛЬЗУЕТСЯ

#### 1. `memory_storage.py` - АКТИВНО ИСПОЛЬЗУЕТСЯ
**Импортируется в:**
- `model_inference_optimized.py` - строка 9: `from .storage import MemoryStorage`

**Классы:**
- `MemoryStorage` - основное хранилище памяти

#### 2. `session_manager.py` - ИСПОЛЬЗУЕТСЯ ВНУТРИ storage/
**Импортируется в:**
- `memory_storage.py` - для управления сессиями
- `session_grouper.py` - для работы с сессиями

**Классы:**
- `SessionManager` - менеджер сессий

#### 3. `memory_models.py` - ИСПОЛЬЗУЕТСЯ ВНУТРИ storage/
**Импортируется в:**
- `memory_storage.py` - для моделей данных
- `session_manager.py` - для моделей сессий

**Классы:**
- `MemoryRegistry` - реестр памяти
- `MemoryStats` - статистика памяти
- `CacheStats` - статистика кэша
- `MemoryEntry` - запись памяти
- `CacheEntry` - запись кэша

#### 4. `session_models.py` - ИСПОЛЬЗУЕТСЯ ВНУТРИ storage/
**Импортируется в:**
- `session_manager.py` - для моделей сессий

**Классы:**
- `SessionRegistry` - реестр сессий
- `SessionInfo` - информация о сессии
- `SessionStats` - статистика сессий

#### 5. `__init__.py` - ЭКСПОРТИРУЕТ ВСЕ КЛАССЫ
**Экспортирует все классы из всех модулей storage/**

## 🎯 КЛЮЧЕВЫЕ ФУНКЦИИ

### filters/ - ОСНОВНЫЕ ФУНКЦИИ
1. **Группировка сессий** - `SessionGrouper` используется во всех основных файлах
2. **Фильтрация копипаста** - `is_copy_paste_content` используется в optimized версии
3. **Определение личных сообщений** - `is_personal_message` используется в optimized версии
4. **Фильтрация релевантности** - используется в RAG системе
5. **Поиск по ключевым словам** - используется в RAG системе

### storage/ - ОСНОВНЫЕ ФУНКЦИИ
1. **Хранение памяти** - `MemoryStorage` используется в optimized версии
2. **Управление сессиями** - `SessionManager` используется внутри storage
3. **Модели данных** - все модели используются внутри storage

## ✅ ВЫВОД

### filters/ - ПОЛНОСТЬЮ АКТИВЕН
- **Все 6 файлов используются**
- **Критически важен для системы**
- **Используется в основных компонентах:**
  - `model_inference_optimized.py`
  - `smart_memory.py`
  - `smart_memory_optimized.py`
  - `rag/compressed_rag_engine.py`
  - `rag/engine.py`

### storage/ - ПОЛНОСТЬЮ АКТИВЕН
- **Все 5 файлов используются**
- **Критически важен для системы**
- **Используется в основных компонентах:**
  - `model_inference_optimized.py`
  - `filters/session_grouper.py`
  - Внутренние зависимости между модулями storage

## 🚫 НЕ УДАЛЯТЬ!

**Обе папки `filters/` и `storage/` являются критически важными компонентами системы и активно используются во всех основных файлах. Удаление любого файла из этих папок приведет к поломке системы!**
