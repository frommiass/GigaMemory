# RAG Architecture Plan

## Структура папок в `src/submit/`:

### 1. **Папка `src/submit/prompts/`**
- `src/submit/prompts/topic_prompts.py` - промпты для каждой темы
- `src/submit/prompts/fallback_prompts.py` - дефолтный промпт

### 2. **Папка `src/submit/questions/`**
- `src/submit/questions/classifier.py` - классификация вопросов по темам
- `src/submit/questions/topics.py` - конфигурация тем и ключевых слов
- `src/submit/questions/confidence.py` - расчет уверенности в классификации

### 3. **Папка `src/submit/filters/`**
- `src/submit/filters/message_cleaner.py` - убирает мусорные сообщения
- `src/submit/filters/session_grouper.py` - группирует сообщения по сессиям
- `src/submit/filters/relevance_filter.py` - находит релевантные сессии
- `src/submit/filters/keyword_matcher.py` - поиск по ключевым словам

### 4. **Папка `src/submit/ranking/`**
- `src/submit/ranking/session_ranker.py` - ранжирование сессий по релевантности
- `src/submit/ranking/scorer.py` - алгоритмы подсчета релевантности

### 5. **Папка `src/submit/rag/`**
- `src/submit/rag/engine.py` - главный RAG движок
- `src/submit/rag/interface.py` - интерфейс для интеграции
- `src/submit/rag/config.py` - настройки

## Этапы реализации:

### **Этап 1: Базовая структура и конфигурация**
1. Создать папки: `prompts/`, `questions/`, `filters/`, `ranking/`, `rag/`
2. Создать `src/submit/rag/config.py` - базовые настройки (threshold, количество сессий)
3. Создать `src/submit/questions/topics.py` - 10-20 тем с ключевыми словами

### **Этап 2: Классификация вопросов**
1. Создать `src/submit/questions/classifier.py` - простой классификатор по ключевым словам
2. Создать `src/submit/questions/confidence.py` - расчет уверенности
3. Протестировать классификацию на примерах

### **Этап 3: Фильтрация и группировка**
1. Создать `src/submit/filters/message_cleaner.py` - убирает мусор
2. Создать `src/submit/filters/session_grouper.py` - группирует по сессиям
3. Создать `src/submit/filters/keyword_matcher.py` - поиск по ключевым словам

### **Этап 4: Поиск релевантных сессий**
1. Создать `src/submit/filters/relevance_filter.py` - находит релевантные сессии
2. Создать `src/submit/ranking/session_ranker.py` - ранжирование сессий
3. Создать `src/submit/ranking/scorer.py` - алгоритмы подсчета релевантности

### **Этап 5: Промпты**
1. Создать `src/submit/prompts/topic_prompts.py` - промпты для каждой темы
2. Создать `src/submit/prompts/fallback_prompts.py` - дефолтный промпт

### **Этап 6: RAG движок**
1. Создать `src/submit/rag/engine.py` - главный движок с логикой фолбэка
2. Создать `src/submit/rag/interface.py` - интерфейс для интеграции

### **Этап 7: Интеграция и тестирование**
1. Обновить `src/submit/model_inference.py` для использования RAG
2. Протестировать на существующих данных
3. Настроить параметры

## Логика работы:

1. **Классификация вопроса** - определить к какой из 10-20 тем относится вопрос
2. **Фильтрация мусора** - убрать нерелевантные сообщения (user+assistant)
3. **Группировка по сессиям** - объединить сообщения в сессии
4. **Поиск релевантных сессий** - по ключевым словам темы найти 1-5 наиболее релевантных сессий
5. **Фильтрация агента** - убрать сообщения ассистента из релевантных сессий
6. **Генерация ответа** - использовать тематический промпт + релевантные сессии пользователя

## Фолбэк:
- Если confidence классификации < 0.7 → используем дефолтный промпт + все сессии пользователя
- Если confidence >= 0.7 → тематический RAG с 1-5 релевантными сессиями

## Шаблон названий:
`{функция}.py` (всегда в конце, единообразно)

