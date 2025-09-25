# src/submit/storage/tests/test_storage.py

import pytest
from typing import List
from models import Message

from ..module import StorageModule
from ..filters.session_grouper import SessionGrouper
from ..message_filter import MessageFilter


class TestStorageModule:
    """Тесты для StorageModule"""
    
    @pytest.fixture
    def storage(self):
        """Создает экземпляр StorageModule для тестов"""
        return StorageModule()
    
    @pytest.fixture
    def sample_messages(self) -> List[Message]:
        """Создает примеры сообщений для тестов"""
        return [
            Message(role="user", content="Привет, я хочу рассказать о своих увлечениях"),
            Message(role="assistant", content="Здравствуйте! Буду рад послушать о ваших увлечениях."),
            Message(role="user", content="Я занимаюсь спортом и люблю читать книги"),
            Message(role="assistant", content="Отлично! Какие виды спорта вас интересуют?"),
            Message(role="user", content="Переведи этот текст: Hello world"),  # Копипаст
            Message(role="assistant", content="Привет мир"),
            Message(role="user", content="Мой любимый спорт - теннис"),
            Message(role="assistant", content="Теннис - отличный выбор!"),
        ]
    
    def test_add_and_get_messages(self, storage, sample_messages):
        """Тест добавления и получения сообщений"""
        dialogue_id = "test_dialogue_1"
        
        # Добавляем сообщения
        result = storage.add_messages(dialogue_id, sample_messages)
        assert result.success
        assert result.metadata['total'] == 8
        assert result.metadata['filtered'] <= 8  # Некоторые могут быть отфильтрованы
        
        # Получаем сообщения
        result = storage.get_messages(dialogue_id)
        assert result.success
        assert isinstance(result.data, list)
        assert len(result.data) > 0
    
    def test_question_filtering(self, storage, sample_messages):
        """Тест фильтрации сообщений для контекста вопроса"""
        dialogue_id = "test_dialogue_2"
        question = "Какими видами спорта ты занимаешься?"
        
        # Добавляем сообщения
        storage.add_messages(dialogue_id, sample_messages)
        
        # Получаем отфильтрованные сессии для вопроса
        result = storage.get_filtered_sessions_for_question(dialogue_id, question)
        assert result.success
        assert isinstance(result.data, dict)
        
        # Проверяем что копипаст отфильтрован
        all_messages = []
        for session_messages in result.data.values():
            all_messages.extend(session_messages)
        
        # Не должно быть сообщения с "Переведи этот текст"
        copypaste_found = any("Переведи" in msg.content for msg in all_messages if msg.role == "user")
        assert not copypaste_found, "Копипаст должен быть отфильтрован"
        
        # Должны остаться сообщения про спорт
        sport_found = any("спорт" in msg.content.lower() for msg in all_messages if msg.role == "user")
        assert sport_found, "Сообщения про спорт должны остаться"
    
    def test_session_grouping(self, storage, sample_messages):
        """Тест группировки сообщений по сессиям"""
        dialogue_id = "test_dialogue_3"
        
        # Добавляем сообщения
        storage.add_messages(dialogue_id, sample_messages)
        
        # Получаем статистику по сессиям
        result = storage.get_session_stats(dialogue_id)
        assert result.success
        assert result.data['total_sessions'] > 0
        assert result.data['total_messages'] > 0
    
    def test_relevance_scoring(self, storage, sample_messages):
        """Тест оценки релевантности сессий"""
        dialogue_id = "test_dialogue_4"
        question = "Расскажи про мои увлечения спортом"
        
        # Добавляем сообщения
        storage.add_messages(dialogue_id, sample_messages)
        
        # Получаем оценки релевантности
        result = storage.get_relevance_scores(dialogue_id, question)
        assert result.success
        assert isinstance(result.data, dict)
        
        # Должна быть хотя бы одна сессия с ненулевым score
        assert any(score > 0 for score in result.data.values())
    
    def test_session_content_extraction(self, storage):
        """Тест извлечения контента сессии для промпта"""
        messages = [
            Message(role="user", content="Я люблю читать книги"),
            Message(role="assistant", content="Какие жанры вам нравятся?"),
            Message(role="user", content="Фантастика и детективы"),
            Message(role="user", content="Переведи: Hello"),  # Копипаст
        ]
        
        # Извлекаем контент
        content = storage.get_session_content_for_prompt(messages)
        
        # Проверяем что копипаст отфильтрован
        assert "Переведи" not in content
        assert "книги" in content
        assert "Фантастика" in content
    
    def test_clear_dialogue(self, storage, sample_messages):
        """Тест очистки диалога"""
        dialogue_id = "test_dialogue_5"
        
        # Добавляем сообщения
        storage.add_messages(dialogue_id, sample_messages)
        
        # Проверяем что сообщения есть
        result = storage.get_messages(dialogue_id)
        assert len(result.data) > 0
        
        # Очищаем диалог
        result = storage.clear(dialogue_id)
        assert result.success
        assert result.metadata['cleared'] == True
        
        # Проверяем что сообщений больше нет
        result = storage.get_messages(dialogue_id)
        assert len(result.data) == 0
    
    def test_memory_stats(self, storage, sample_messages):
        """Тест получения статистики памяти"""
        dialogue_id = "test_dialogue_6"
        
        # Добавляем сообщения
        storage.add_messages(dialogue_id, sample_messages)
        
        # Получаем статистику
        result = storage.get_memory_stats(dialogue_id)
        assert result.success
        assert 'messages_count' in result.data
        assert 'sessions_count' in result.data
        assert 'personal_messages' in result.data
    
    def test_cache_functionality(self, storage):
        """Тест работы кэширования"""
        dialogue_id = "test_dialogue_7"
        question = "Тестовый вопрос"
        messages = [
            Message(role="user", content="Тестовое сообщение для кэширования"),
            Message(role="assistant", content="Ответ"),
        ]
        
        # Первый вызов - без кэша
        filtered1 = storage.filter_for_question_context(messages, question)
        
        # Второй вызов - должен использовать кэш
        filtered2 = storage.filter_for_question_context(messages, question)
        
        # Результаты должны быть идентичны
        assert len(filtered1) == len(filtered2)
        assert all(m1.content == m2.content for m1, m2 in zip(filtered1, filtered2))


class TestSessionGrouper:
    """Тесты для SessionGrouper"""
    
    @pytest.fixture
    def grouper(self):
        """Создает экземпляр SessionGrouper для тестов"""
        return SessionGrouper()
    
    def test_session_increment(self, grouper):
        """Тест увеличения счетчика сессий"""
        dialogue_id = "test_dialogue"
        
        # Первая сессия
        session1 = grouper.increment_session(dialogue_id)
        assert session1 == 1
        
        # Вторая сессия
        session2 = grouper.increment_session(dialogue_id)
        assert session2 == 2
        
        # Проверяем текущую сессию
        current = grouper.get_current_session(dialogue_id)
        assert current == 2
    
    def test_session_registration(self, grouper):
        """Тест регистрации сессий"""
        dialogue_id = "test_dialogue"
        session_id = "custom_session_1"
        
        # Регистрируем сессию
        grouper.register_session(dialogue_id, session_id)
        
        # Проверяем регистрацию
        assert grouper.is_session_registered(dialogue_id, session_id)
        assert session_id in grouper.get_session_ids(dialogue_id)
    
    def test_group_messages(self, grouper):
        """Тест группировки сообщений по сессиям"""
        dialogue_id = "test_dialogue"
        messages = [
            Message(role="user", content="Первое сообщение"),
            Message(role="assistant", content="Ответ 1"),
            Message(role="user", content="Второе сообщение"),
            Message(role="assistant", content="Ответ 2"),
        ]
        
        # Группируем сообщения
        sessions = grouper.group_messages_by_sessions(messages, dialogue_id)
        
        # Проверяем результат
        assert len(sessions) > 0
        assert all(isinstance(msgs, list) for msgs in sessions.values())
        
        # Проверяем что сообщения сгруппированы правильно
        for session_messages in sessions.values():
            # Должны быть и user и assistant сообщения
            roles = [msg.role for msg in session_messages]
            assert "user" in roles
    
    def test_session_stats(self, grouper):
        """Тест получения статистики сессий"""
        dialogue_id = "test_dialogue"
        messages = [
            Message(role="user", content="Сообщение 1"),
            Message(role="assistant", content="Ответ 1"),
            Message(role="user", content="Сообщение 2"),
            Message(role="assistant", content="Ответ 2"),
        ]
        
        # Группируем и получаем статистику
        grouper.group_messages_by_sessions(messages, dialogue_id)
        stats = grouper.get_session_stats(dialogue_id)
        
        # Проверяем статистику
        assert stats['dialogue_id'] == dialogue_id
        assert stats['total_sessions'] > 0
        assert stats['total_messages'] > 0
        assert stats['avg_messages_per_session'] > 0
    
    def test_clear_sessions(self, grouper):
        """Тест очистки сессий диалога"""
        dialogue_id = "test_dialogue"
        
        # Создаем сессии
        grouper.increment_session(dialogue_id)
        grouper.register_session(dialogue_id, "session_1")
        
        # Очищаем
        grouper.clear_dialogue_sessions(dialogue_id)
        
        # Проверяем очистку
        assert grouper.get_current_session(dialogue_id) == 0
        assert grouper.get_session_count(dialogue_id) == 0
        assert len(grouper.get_session_ids(dialogue_id)) == 0


class TestMessageFilter:
    """Тесты для MessageFilter"""
    
    @pytest.fixture
    def filter(self):
        """Создает экземпляр MessageFilter для тестов"""
        return MessageFilter()
    
    def test_filter_copypaste(self, filter):
        """Тест фильтрации копипаста"""
        messages = [
            Message(role="user", content="Мой обычный текст"),
            Message(role="user", content="Переведи этот текст: Hello world"),
            Message(role="user", content="Исправь ошибки в коде: print('test')"),
            Message(role="user", content="Я думаю что это интересно"),
        ]
        
        # Фильтруем
        filtered = filter.filter_messages(messages)
        
        # Проверяем что копипаст удален
        contents = [msg.content for msg in filtered]
        assert not any("Переведи" in c for c in contents)
        assert not any("Исправь" in c for c in contents)
        assert any("обычный текст" in c for c in contents)
        assert any("интересно" in c for c in contents)
    
    def test_filter_technical(self, filter):
        """Тест фильтрации технического контента"""
        messages = [
            Message(role="user", content="Я работаю программистом"),
            Message(role="user", content="def function(): return 42"),
            Message(role="user", content="SELECT * FROM table WHERE id = 1"),
            Message(role="user", content="Мне нравится читать книги"),
        ]
        
        # Фильтруем
        filtered = filter.filter_messages(messages)
        
        # Проверяем что технический контент удален
        contents = [msg.content for msg in filtered if msg.role == "user"]
        assert not any("def function" in c for c in contents)
        assert not any("SELECT" in c for c in contents)
        assert any("программистом" in c for c in contents)
        assert any("книги" in c for c in contents)
    
    def test_message_analysis(self, filter):
        """Тест анализа сообщений"""
        messages = [
            Message(role="user", content="Личное сообщение"),
            Message(role="user", content="Переведи: Hello"),
            Message(role="user", content="def code(): pass"),
            Message(role="assistant", content="Ответ"),
        ]
        
        # Анализируем
        analysis = filter.get_message_analysis(messages)
        
        # Проверяем результаты анализа
        assert analysis['total_messages'] == 4
        assert analysis['user_messages'] == 3
        assert analysis['copypaste_messages'] >= 1
        assert analysis['technical_messages'] >= 1
        assert 'filter_ratio' in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])