#!/usr/bin/env python3
"""
Тестирование системы с мок-компонентами
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import Message
import logging
import time
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockModelInference:
    """Мок-версия ModelInference"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"🤖 MockModelInference инициализирован с моделью: {model_path}")
    
    def inference(self, messages: List[Message]) -> str:
        """Генерирует мок-ответ"""
        # Простая логика для демонстрации
        system_message = None
        for msg in messages:
            if msg.role == 'system':
                system_message = msg.content
                break
        
        if system_message:
            if "работа" in system_message.lower():
                return "Пользователь работает программистом в компании Яндекс."
            elif "семья" in system_message.lower():
                return "У пользователя есть жена Мария и дочь Анна 5 лет."
            elif "живет" in system_message.lower():
                return "Пользователь живет в Москве, в районе Сокольники."
            elif "увлечения" in system_message.lower():
                return "Пользователь увлекается фотографией."
            elif "животные" in system_message.lower():
                return "У пользователя есть собака по кличке Рекс."
            elif "изучает" in system_message.lower():
                return "Пользователь изучает Python и машинное обучение."
            elif "жена" in system_message.lower():
                return "Жена пользователя работает учителем в школе."
            elif "дочь" in system_message.lower():
                return "Дочери пользователя 5 лет."
        
        return "Я нашел информацию в памяти пользователя."

class MockEmbeddingEngine:
    """Мок-версия EmbeddingEngine"""
    
    def __init__(self):
        print("🔍 MockEmbeddingEngine инициализирован")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Создает мок-эмбеддинги"""
        embeddings = []
        for text in texts:
            # Простой хэш для создания "уникальных" эмбеддингов
            hash_val = hash(text) % 1000
            embedding = [hash_val / 1000.0] * 384  # Стандартный размер
            embeddings.append(embedding)
        return embeddings

class MockVectorStore:
    """Мок-версия VectorStore"""
    
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        print("📚 MockVectorStore инициализирован")
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict] = None):
        """Добавляет документы"""
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{len(self.documents)}"
            self.documents[doc_id] = doc
            self.embeddings[doc_id] = emb
        print(f"📝 Добавлено {len(documents)} документов")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Поиск похожих документов"""
        # Простой поиск по косинусному сходству
        results = []
        for doc_id, emb in self.embeddings.items():
            # Простое скалярное произведение
            similarity = sum(a * b for a, b in zip(query_embedding, emb)) / len(emb)
            results.append({
                'document': self.documents[doc_id],
                'similarity': similarity,
                'metadata': {'doc_id': doc_id}
            })
        
        # Сортируем по убыванию схожести
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

class MockFactExtractor:
    """Мок-версия FactExtractor"""
    
    def __init__(self):
        print("📝 MockFactExtractor инициализирован")
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List:
        """Извлекает мок-факты"""
        facts = []
        
        # Простое извлечение фактов по ключевым словам
        if "программист" in text.lower():
            facts.append({
                'type': 'work',
                'content': 'Работает программистом',
                'confidence': 0.9
            })
        
        if "яндекс" in text.lower():
            facts.append({
                'type': 'work',
                'content': 'Работает в компании Яндекс',
                'confidence': 0.8
            })
        
        if "жена" in text.lower():
            facts.append({
                'type': 'family',
                'content': 'Есть жена',
                'confidence': 0.9
            })
        
        if "дочь" in text.lower():
            facts.append({
                'type': 'family',
                'content': 'Есть дочь',
                'confidence': 0.9
            })
        
        if "москва" in text.lower():
            facts.append({
                'type': 'location',
                'content': 'Живет в Москве',
                'confidence': 0.8
            })
        
        if "фотографией" in text.lower():
            facts.append({
                'type': 'hobby',
                'content': 'Увлекается фотографией',
                'confidence': 0.7
            })
        
        print(f"📊 Извлечено {len(facts)} фактов из текста")
        return facts

class MockCompressor:
    """Мок-версия Compressor"""
    
    def __init__(self):
        print("🗜️ MockCompressor инициализирован")
    
    def compress(self, text: str) -> Dict[str, Any]:
        """Сжимает текст"""
        # Простое сжатие - берем первые 100 символов
        compressed = text[:100] + "..." if len(text) > 100 else text
        ratio = len(compressed) / len(text) if len(text) > 0 else 1.0
        
        return {
            'compressed_text': compressed,
            'compression_ratio': ratio,
            'original_length': len(text),
            'compressed_length': len(compressed)
        }

class MockRAGEngine:
    """Мок-версия RAG Engine"""
    
    def __init__(self):
        self.embedding_engine = MockEmbeddingEngine()
        self.vector_store = MockVectorStore()
        self.fact_extractor = MockFactExtractor()
        self.compressor = MockCompressor()
        print("🎯 MockRAGEngine инициализирован")
    
    def index_dialogue_compressed(self, dialogue_id: str, sessions: Dict[str, List[Message]]) -> Dict[str, Any]:
        """Индексирует диалог"""
        print(f"📚 Индексирование диалога {dialogue_id}")
        
        documents = []
        embeddings = []
        facts_count = 0
        compression_stats = {'total_ratio': 0.0, 'count': 0}
        
        for session_id, messages in sessions.items():
            # Объединяем сообщения сессии
            session_text = " ".join([msg.content for msg in messages])
            
            # Сжимаем текст
            compression_result = self.compressor.compress(session_text)
            compressed_text = compression_result['compressed_text']
            
            # Создаем эмбеддинг
            embedding = self.embedding_engine.encode([compressed_text])[0]
            
            # Добавляем в хранилище
            documents.append(compressed_text)
            embeddings.append(embedding)
            
            # Извлекаем факты
            facts = self.fact_extractor.extract_facts_from_text(session_text, session_id, dialogue_id)
            facts_count += len(facts)
            
            # Обновляем статистику сжатия
            compression_stats['total_ratio'] += compression_result['compression_ratio']
            compression_stats['count'] += 1
        
        # Добавляем документы в векторное хранилище
        self.vector_store.add_documents(documents, embeddings)
        
        # Вычисляем средний коэффициент сжатия
        avg_compression_ratio = compression_stats['total_ratio'] / compression_stats['count'] if compression_stats['count'] > 0 else 1.0
        
        return {
            'sessions_indexed': len(sessions),
            'facts_extracted': facts_count,
            'compression': {
                'ratio': avg_compression_ratio,
                'total_sessions': compression_stats['count']
            }
        }
    
    def process_question(self, question: str, dialogue_id: str, memory: List[Message]) -> tuple:
        """Обрабатывает вопрос"""
        print(f"❓ Обработка вопроса: {question}")
        
        # Создаем эмбеддинг вопроса
        query_embedding = self.embedding_engine.encode([question])[0]
        
        # Ищем похожие документы
        similar_docs = self.vector_store.search(query_embedding, top_k=3)
        
        # Формируем контекст
        context_parts = []
        for doc in similar_docs:
            context_parts.append(f"- {doc['document']}")
        
        context = "\n".join(context_parts)
        
        # Создаем промпт
        prompt = f"""Ты - помощник с доступом к памяти пользователя. 
Используй следующую информацию для ответа на вопрос:

КОНТЕКСТ ИЗ ПАМЯТИ:
{context}

ВОПРОС: {question}

Ответь кратко и по существу, используя информацию из контекста."""
        
        return prompt, {'similar_docs': len(similar_docs)}

class MockOptimizedSmartMemory:
    """Мок-версия OptimizedSmartMemory"""
    
    def __init__(self, model_path: str = "/mock/model/path"):
        print("🧠 MockOptimizedSmartMemory инициализирована")
        
        self.model = MockModelInference(model_path)
        self.rag_engine = MockRAGEngine()
        
        # Статистика
        self.metrics = {
            'total_queries': 0,
            'avg_query_time': 0.0,
            'cache_hit_rate': 0.0,
            'compression_ratio': 0.0,
            'facts_per_dialogue': 0.0
        }
        
        print("✅ Система готова к работе")
    
    def process_dialogue_optimized(self, dialogue_id: str, messages: List[Message]) -> Dict[str, Any]:
        """Обрабатывает диалог"""
        print(f"🔄 Обработка диалога {dialogue_id}")
        
        start_time = time.time()
        
        # Группируем сообщения по сессиям (простая группировка)
        sessions = {}
        current_session = "session_1"
        sessions[current_session] = messages
        
        # Индексируем через RAG
        stats = self.rag_engine.index_dialogue_compressed(dialogue_id, sessions)
        
        # Добавляем общую статистику
        stats.update({
            'dialogue_id': dialogue_id,
            'messages_count': len(messages),
            'sessions_count': len(sessions),
            'processing_time': time.time() - start_time,
            'cache_used': False
        })
        
        # Обновляем метрики
        self.metrics['compression_ratio'] = stats['compression']['ratio']
        self.metrics['facts_per_dialogue'] = stats['facts_extracted']
        
        return stats
    
    def answer_question_optimized(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос"""
        print(f"❓ Ответ на вопрос: {question}")
        
        start_time = time.time()
        
        # Обрабатываем вопрос через RAG
        prompt, metadata = self.rag_engine.process_question(question, dialogue_id, [])
        
        # Создаем сообщение для модели
        context_message = Message('system', prompt)
        
        # Генерируем ответ
        answer = self.model.inference([context_message])
        
        # Обновляем метрики
        query_time = time.time() - start_time
        self.metrics['total_queries'] += 1
        self.metrics['avg_query_time'] = (
            0.1 * query_time + 0.9 * self.metrics['avg_query_time']
        )
        
        return answer.strip()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы"""
        return {
            'metrics': self.metrics,
            'cache': {'hit_rate': 0.85, 'total_entries': 150},
            'queues': {'embedding_queue': 0, 'fact_queue': 0, 'compression_queue': 0},
            'rag': {'total_documents': 8, 'avg_similarity': 0.75},
            'compression': {'avg_ratio': self.metrics['compression_ratio']}
        }
    
    def save_state(self, save_dir: str = "./test_state"):
        """Сохраняет состояние"""
        print(f"💾 Сохранение состояния в {save_dir}")
        # В реальной системе здесь было бы сохранение на диск
        print("✅ Состояние сохранено")

def test_mock_system():
    """Тестирует мок-систему"""
    
    print("🚀 Запуск тестирования мок-системы SmartMemory")
    print("=" * 60)
    
    try:
        # 1. Инициализация системы
        print("📋 Инициализация мок-системы...")
        memory = MockOptimizedSmartMemory()
        
        # 2. Тестовые сообщения
        print("\n📝 Подготовка тестовых данных...")
        test_messages = [
            Message(role="user", content="Привет! Меня зовут Алексей, я работаю программистом в компании Яндекс."),
            Message(role="user", content="У меня есть жена Мария и дочь Анна, которой 5 лет."),
            Message(role="user", content="Я живу в Москве, в районе Сокольники. Работаю удаленно."),
            Message(role="user", content="Моя жена работает учителем в школе. Она очень любит свою работу."),
            Message(role="user", content="У нас есть собака по кличке Рекс. Он очень дружелюбный."),
            Message(role="user", content="Я увлекаюсь фотографией и часто фотографирую семью на выходных."),
            Message(role="user", content="Наша дочь ходит в детский сад и очень любит рисовать."),
            Message(role="user", content="Я работаю с Python и машинным обучением. Недавно изучаю нейросети."),
        ]
        
        dialogue_id = "test_dialogue_1"
        print(f"✅ Подготовлено {len(test_messages)} тестовых сообщений")
        
        # 3. Обработка диалога
        print(f"\n🔄 Обработка диалога {dialogue_id}...")
        stats = memory.process_dialogue_optimized(dialogue_id, test_messages)
        
        print("📊 Статистика обработки:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
        
        # 4. Тестовые вопросы
        print(f"\n❓ Тестирование вопросов...")
        test_questions = [
            "Расскажи о работе пользователя",
            "Какая у пользователя семья?",
            "Где живет пользователь?",
            "Какие у пользователя увлечения?",
            "Расскажи о домашних животных",
            "Что изучает пользователь?",
            "Где работает жена пользователя?",
            "Сколько лет дочери пользователя?"
        ]
        
        # 5. Генерация ответов
        print("\n🤖 Генерация ответов:")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Вопрос: {question}")
            try:
                answer = memory.answer_question_optimized(dialogue_id, question)
                print(f"   Ответ: {answer}")
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        # 6. Статистика системы
        print(f"\n📈 Общая статистика системы:")
        print("-" * 40)
        system_stats = memory.get_system_stats()
        
        for category, stats in system_stats.items():
            print(f"\n{category.upper()}:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        # 7. Сохранение состояния
        print(f"\n💾 Сохранение состояния...")
        memory.save_state("./test_state")
        
        print(f"\n🎉 Тестирование завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mock_system()
