#!/usr/bin/env python3
"""
Тест для EmbeddingsModule - проверка всех методов интерфейса IEmbeddingEngine
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
from submit.modules.embeddings.module import EmbeddingsModule

def test_embeddings_module():
    """Тестирует все методы EmbeddingsModule"""
    
    # Конфигурация
    config = {
        'model_name': 'cointegrated/rubert-tiny2',
        'device': 'cpu',
        'batch_size': 32,
        'use_cache': True
    }
    
    # Создаем модуль
    embeddings = EmbeddingsModule(config)
    
    # Тестовые данные
    dialogue_id = "test_dialogue_1"
    texts = ["Привет, как дела?", "Расскажи о погоде", "Что нового?"]
    
    # Создаем тестовые сообщения для сессий
    messages1 = [
        Message(role="user", content="Привет, как дела?", session_id="session_1"),
        Message(role="assistant", content="Привет! У меня все хорошо, спасибо!", session_id="session_1")
    ]
    messages2 = [
        Message(role="user", content="Расскажи о погоде", session_id="session_2"),
        Message(role="assistant", content="Сегодня солнечно и тепло", session_id="session_2")
    ]
    
    sessions = {
        "session_1": messages1,
        "session_2": messages2
    }
    
    print("🧪 Тестирование EmbeddingsModule...")
    
    # 1. Тест encode_texts
    print("\n1. Тестируем encode_texts...")
    result = embeddings.encode_texts(texts)
    assert result.success, f"encode_texts failed: {result.error}"
    assert result.data is not None, "encode_texts returned None data"
    assert len(result.data) == len(texts), f"Expected {len(texts)} embeddings, got {len(result.data)}"
    print(f"   ✅ encode_texts: {len(result.data)} эмбеддингов создано")
    
    # 2. Тест index_dialogue
    print("\n2. Тестируем index_dialogue...")
    result = embeddings.index_dialogue(dialogue_id, sessions)
    assert result.success, f"index_dialogue failed: {result.error}"
    assert result.metadata['sessions_indexed'] == len(sessions), "Не все сессии проиндексированы"
    print(f"   ✅ index_dialogue: {result.metadata['sessions_indexed']} сессий проиндексировано")
    
    # 3. Тест vector_search (после индексации)
    print("\n3. Тестируем vector_search...")
    query = "как дела"
    result = embeddings.vector_search(query, dialogue_id, top_k=3)
    assert result.success, f"vector_search failed: {result.error}"
    print(f"   ✅ vector_search: найдено {len(result.data)} результатов")
    
    # 4. Тест vector_search для несуществующего диалога
    print("\n4. Тестируем vector_search для несуществующего диалога...")
    result = embeddings.vector_search(query, "nonexistent_dialogue", top_k=3)
    assert not result.success, "vector_search should fail for nonexistent dialogue"
    assert "not indexed" in result.error, "Error message should mention 'not indexed'"
    print(f"   ✅ vector_search: корректно обработал несуществующий диалог")
    
    # 5. Тест с пустым списком текстов
    print("\n5. Тестируем encode_texts с пустым списком...")
    result = embeddings.encode_texts([])
    assert result.success, f"encode_texts failed with empty list: {result.error}"
    assert len(result.data) == 0, "Empty list should return empty embeddings"
    print(f"   ✅ encode_texts: корректно обработал пустой список")
    
    # 6. Тест с пустыми сессиями
    print("\n6. Тестируем index_dialogue с пустыми сессиями...")
    empty_sessions = {}
    result = embeddings.index_dialogue("empty_dialogue", empty_sessions)
    assert result.success, f"index_dialogue failed with empty sessions: {result.error}"
    assert result.metadata['sessions_indexed'] == 0, "Empty sessions should return 0 indexed"
    print(f"   ✅ index_dialogue: корректно обработал пустые сессии")
    
    print("\n🎉 Все тесты EmbeddingsModule прошли успешно!")
    print("\n📋 Проверочный чеклист:")
    print("✅ IEmbeddingEngine.encode_texts - работает")
    print("✅ IEmbeddingEngine.vector_search - работает") 
    print("✅ IEmbeddingEngine.index_dialogue - работает")
    print("✅ Обработка ошибок - работает")
    print("✅ Граничные случаи - работают")

if __name__ == "__main__":
    test_embeddings_module()
