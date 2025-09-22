#!/usr/bin/env python3
"""
Упрощенный тест RAG компонентов без внешних зависимостей
"""
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Создаем простую модель Message
class Message:
    def __init__(self, role: str, content: str, session_id: str = None):
        self.role = role
        self.content = content
        self.session_id = session_id

# Мокаем models модуль
sys.modules['models'] = type('models', (), {'Message': Message})()

def test_rag_components():
    """Тестирует RAG компоненты по отдельности"""
    print("🧪 Тестирование RAG компонентов...")
    
    # Тест 1: Классификатор вопросов
    print("\n1️⃣ Тестирование классификатора вопросов:")
    try:
        from submit.questions.classifier import QuestionClassifier
        classifier = QuestionClassifier()
        
        test_questions = [
            "Как меня зовут?",
            "Какие у меня животные?",
            "Где я работаю?",
            "Что я делаю в свободное время?"
        ]
        
        for question in test_questions:
            topic, confidence = classifier.classify_question(question)
            print(f"   '{question}' → Тема: {topic}, Уверенность: {confidence:.2f}")
        
        print("   ✅ Классификатор работает!")
    except Exception as e:
        print(f"   ❌ Ошибка в классификаторе: {e}")
    
    # Тест 2: Фильтр сообщений
    print("\n2️⃣ Тестирование фильтра сообщений:")
    try:
        from submit.filters.message_cleaner import is_personal_message
        
        test_messages = [
            "Привет, меня зовут Иван",
            "У меня есть кот",
            "def hello(): print('world')",  # Технический код
            "Переведи этот текст на английский",  # Копипаст
            "Я работаю программистом"
        ]
        
        for msg in test_messages:
            is_personal = is_personal_message(msg)
            print(f"   '{msg[:30]}...' → Личное: {is_personal}")
        
        print("   ✅ Фильтр сообщений работает!")
    except Exception as e:
        print(f"   ❌ Ошибка в фильтре сообщений: {e}")
    
    # Тест 3: Группировщик сессий
    print("\n3️⃣ Тестирование группировщика сессий:")
    try:
        from submit.filters.session_grouper import SessionGrouper
        
        test_messages = [
            Message(role="user", content="Привет, меня зовут Иван", session_id="1"),
            Message(role="user", content="У меня есть кот", session_id="1"),
            Message(role="user", content="Я работаю программистом", session_id="2"),
        ]
        
        grouper = SessionGrouper()
        sessions = grouper.group_messages_by_sessions(test_messages, "test_dialogue")
        
        print(f"   Количество сессий: {len(sessions)}")
        for session_id, messages in sessions.items():
            print(f"   Сессия {session_id}: {len(messages)} сообщений")
        
        print("   ✅ Группировщик сессий работает!")
    except Exception as e:
        print(f"   ❌ Ошибка в группировщике сессий: {e}")
    
    # Тест 4: Поиск по ключевым словам
    print("\n4️⃣ Тестирование поиска по ключевым словам:")
    try:
        from submit.filters.keyword_matcher import KeywordMatcher
        
        test_messages = [
            Message(role="user", content="У меня есть кот по имени Барсик", session_id="1"),
            Message(role="user", content="Мой кот любит играть", session_id="2"),
            Message(role="user", content="Я работаю программистом", session_id="3"),
        ]
        
        sessions = {"1": [test_messages[0]], "2": [test_messages[1]], "3": [test_messages[2]]}
        
        matcher = KeywordMatcher()
        pet_sessions = matcher.find_sessions_by_topic(sessions, "pets")
        
        print(f"   Сессии по теме 'pets': {list(pet_sessions.keys())}")
        
        print("   ✅ Поиск по ключевым словам работает!")
    except Exception as e:
        print(f"   ❌ Ошибка в поиске по ключевым словам: {e}")
    
    # Тест 5: Система скоринга
    print("\n5️⃣ Тестирование системы скоринга:")
    try:
        from submit.ranking.scorer import RelevanceScorer
        
        scorer = RelevanceScorer()
        score = scorer.calculate_session_score(
            "У меня есть кот по имени Барсик",
            "Какие у меня животные?",
            {"кот", "животные"}
        )
        
        print(f"   Счет релевантности: {score:.3f}")
        
        print("   ✅ Система скоринга работает!")
    except Exception as e:
        print(f"   ❌ Ошибка в системе скоринга: {e}")
    
    # Тест 6: Ранжирование сессий
    print("\n6️⃣ Тестирование ранжирования сессий:")
    try:
        from submit.ranking.session_ranker import SessionRanker
        
        test_messages = [
            Message(role="user", content="У меня есть кот по имени Барсик", session_id="1"),
            Message(role="user", content="Мой кот любит играть", session_id="2"),
            Message(role="user", content="Я работаю программистом", session_id="3"),
        ]
        
        sessions = {"1": [test_messages[0]], "2": [test_messages[1]], "3": [test_messages[2]]}
        
        ranker = SessionRanker()
        ranking = ranker.rank_sessions("Какие у меня животные?", sessions)
        
        print(f"   Ранжированных сессий: {len(ranking)}")
        for session_id, score in ranking:
            print(f"   Сессия {session_id}: {score:.3f}")
        
        print("   ✅ Ранжирование сессий работает!")
    except Exception as e:
        print(f"   ❌ Ошибка в ранжировании сессий: {e}")
    
    print("\n✅ Все тесты завершены!")


if __name__ == "__main__":
    test_rag_components()
