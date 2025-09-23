#!/usr/bin/env python3
"""
Полная демонстрация системы GigaMemory с генерацией промптов
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

class GigaMemoryDemo:
    """Демонстрация полной системы GigaMemory"""
    
    def __init__(self):
        print("🚀 Инициализация демонстрации GigaMemory")
        print("=" * 60)
        
        # Инициализируем компоненты
        self.embedding_engine = MockEmbeddingEngine()
        self.vector_store = MockVectorStore()
        self.fact_extractor = MockFactExtractor()
        self.compressor = MockCompressor()
        self.prompt_generator = AdvancedPromptGenerator()
        
        # База фактов
        self.fact_database = {}
        
        # Статистика
        self.stats = {
            'dialogues_processed': 0,
            'total_facts_extracted': 0,
            'total_queries': 0,
            'avg_compression_ratio': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print("✅ Система инициализирована")
    
    def process_dialogue(self, dialogue_id: str, messages: List[Message]) -> Dict[str, Any]:
        """Обрабатывает диалог полным циклом"""
        print(f"\n🔄 Обработка диалога: {dialogue_id}")
        print("-" * 40)
        
        start_time = time.time()
        
        # 1. Группировка по сессиям
        sessions = self._group_messages_by_sessions(messages)
        print(f"📊 Создано сессий: {len(sessions)}")
        
        # 2. Обработка каждой сессии
        total_facts = 0
        compression_ratios = []
        
        for session_id, session_messages in sessions.items():
            print(f"\n📝 Обработка сессии: {session_id}")
            
            # Объединяем сообщения сессии
            session_text = " ".join([msg.content for msg in session_messages])
            print(f"   Текст сессии: {session_text[:100]}...")
            
            # Сжатие
            compression_result = self.compressor.compress(session_text)
            compression_ratios.append(compression_result['compression_ratio'])
            print(f"   Сжатие: {compression_result['compression_ratio']:.2f}")
            
            # Создание эмбеддинга
            embedding = self.embedding_engine.encode([session_text])[0]
            print(f"   Эмбеддинг создан: {len(embedding)} измерений")
            
            # Извлечение фактов
            facts = self.fact_extractor.extract_facts_from_text(session_text, session_id, dialogue_id)
            total_facts += len(facts)
            print(f"   Извлечено фактов: {len(facts)}")
            
            # Сохранение в векторное хранилище
            self.vector_store.add_documents([session_text], [embedding])
            
            # Сохранение фактов
            if dialogue_id not in self.fact_database:
                self.fact_database[dialogue_id] = []
            self.fact_database[dialogue_id].extend(facts)
        
        # 3. Обновление статистики
        processing_time = time.time() - start_time
        avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 1.0
        
        self.stats['dialogues_processed'] += 1
        self.stats['total_facts_extracted'] += total_facts
        self.stats['avg_compression_ratio'] = avg_compression
        
        result = {
            'dialogue_id': dialogue_id,
            'sessions_count': len(sessions),
            'facts_extracted': total_facts,
            'compression_ratio': avg_compression,
            'processing_time': processing_time,
            'documents_indexed': len(sessions)
        }
        
        print(f"\n✅ Диалог обработан за {processing_time:.3f}с")
        return result
    
    def answer_question(self, dialogue_id: str, question: str) -> str:
        """Отвечает на вопрос используя всю систему"""
        print(f"\n❓ Обработка вопроса: {question}")
        print("-" * 40)
        
        start_time = time.time()
        
        # 1. Поиск релевантных документов
        query_embedding = self.embedding_engine.encode([question])[0]
        similar_docs = self.vector_store.search(query_embedding, top_k=3)
        
        print(f"🔍 Найдено релевантных документов: {len(similar_docs)}")
        for i, doc in enumerate(similar_docs, 1):
            print(f"   {i}. Схожесть: {doc['similarity']:.3f}")
            print(f"      Текст: {doc['document'][:80]}...")
        
        # 2. Получение фактов
        relevant_facts = self.fact_database.get(dialogue_id, [])
        print(f"📊 Доступно фактов: {len(relevant_facts)}")
        
        # 3. Генерация промпта
        memory_data = [doc['document'] for doc in similar_docs]
        prompt = self.prompt_generator.generate_enhanced_prompt(question, memory_data)
        
        print(f"\n📝 Сгенерированный промпт:")
        print("=" * 50)
        print(prompt)
        print("=" * 50)
        
        # 4. Генерация ответа (мок)
        answer = self._generate_mock_answer(question, similar_docs, relevant_facts)
        
        # 5. Обновление статистики
        query_time = time.time() - start_time
        self.stats['total_queries'] += 1
        
        print(f"\n🤖 Ответ: {answer}")
        print(f"⏱️ Время обработки: {query_time:.3f}с")
        
        return answer
    
    def _group_messages_by_sessions(self, messages: List[Message]) -> Dict[str, List[Message]]:
        """Группирует сообщения по сессиям"""
        sessions = {}
        current_session = "session_1"
        
        for msg in messages:
            if current_session not in sessions:
                sessions[current_session] = []
            sessions[current_session].append(msg)
        
        return sessions
    
    def _generate_mock_answer(self, question: str, similar_docs: List[Dict], facts: List[Dict]) -> str:
        """Генерирует мок-ответ на основе контекста"""
        question_lower = question.lower()
        
        # Простая логика ответов на основе контекста
        if "работа" in question_lower:
            return "Пользователь работает программистом в компании Яндекс, специализируется на Python и машинном обучении."
        elif "семья" in question_lower:
            return "У пользователя есть жена Мария, которая работает учителем в школе, и дочь Анна 5 лет."
        elif "живет" in question_lower:
            return "Пользователь живет в Москве, в районе Сокольники, работает удаленно."
        elif "увлечения" in question_lower:
            return "Пользователь увлекается фотографией и часто фотографирует семью на выходных."
        elif "животные" in question_lower:
            return "У пользователя есть собака по кличке Рекс, которая очень дружелюбная."
        elif "изучает" in question_lower:
            return "Пользователь изучает Python, машинное обучение и недавно начал изучать нейросети."
        elif "жена" in question_lower:
            return "Жена пользователя Мария работает учителем в школе и очень любит свою работу."
        elif "дочь" in question_lower:
            return "Дочери пользователя Анне 5 лет, она ходит в детский сад и любит рисовать."
        else:
            return "На основе доступной информации могу рассказать о работе, семье, местоположении и увлечениях пользователя."
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы"""
        return {
            'processing': self.stats,
            'storage': {
                'total_documents': len(self.vector_store.documents),
                'total_facts': sum(len(facts) for facts in self.fact_database.values()),
                'dialogues_count': len(self.fact_database)
            },
            'performance': {
                'avg_query_time': 0.15,  # Мок-значение
                'cache_hit_rate': 0.85,   # Мок-значение
                'compression_efficiency': self.stats['avg_compression_ratio']
            }
        }

class MockEmbeddingEngine:
    """Мок-версия EmbeddingEngine"""
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            hash_val = hash(text) % 1000
            embedding = [hash_val / 1000.0] * 384
            embeddings.append(embedding)
        return embeddings

class MockVectorStore:
    """Мок-версия VectorStore"""
    
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]]):
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{len(self.documents)}"
            self.documents[doc_id] = doc
            self.embeddings[doc_id] = emb
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        results = []
        for doc_id, emb in self.embeddings.items():
            similarity = sum(a * b for a, b in zip(query_embedding, emb)) / len(emb)
            results.append({
                'document': self.documents[doc_id],
                'similarity': similarity,
                'metadata': {'doc_id': doc_id}
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

class MockFactExtractor:
    """Мок-версия FactExtractor"""
    
    def extract_facts_from_text(self, text: str, session_id: str, dialogue_id: str) -> List[Dict]:
        facts = []
        text_lower = text.lower()
        
        if "программист" in text_lower:
            facts.append({'type': 'work', 'content': 'Работает программистом', 'confidence': 0.9})
        if "яндекс" in text_lower:
            facts.append({'type': 'work', 'content': 'Работает в Яндекс', 'confidence': 0.8})
        if "жена" in text_lower:
            facts.append({'type': 'family', 'content': 'Есть жена', 'confidence': 0.9})
        if "дочь" in text_lower:
            facts.append({'type': 'family', 'content': 'Есть дочь', 'confidence': 0.9})
        if "москва" in text_lower:
            facts.append({'type': 'location', 'content': 'Живет в Москве', 'confidence': 0.8})
        if "фотографией" in text_lower:
            facts.append({'type': 'hobby', 'content': 'Увлекается фотографией', 'confidence': 0.7})
        if "собака" in text_lower:
            facts.append({'type': 'pets', 'content': 'Есть собака', 'confidence': 0.8})
        
        return facts

class MockCompressor:
    """Мок-версия Compressor"""
    
    def compress(self, text: str) -> Dict[str, Any]:
        compressed = text[:100] + "..." if len(text) > 100 else text
        ratio = len(compressed) / len(text) if len(text) > 0 else 1.0
        
        return {
            'compressed_text': compressed,
            'compression_ratio': ratio,
            'original_length': len(text),
            'compressed_length': len(compressed)
        }

class AdvancedPromptGenerator:
    """Продвинутый генератор промптов"""
    
    def __init__(self):
        self.fact_categories = {
            'work': ['программист', 'работа', 'компания', 'яндекс', 'python'],
            'family': ['жена', 'дочь', 'семья', 'мария', 'анна'],
            'location': ['москва', 'живет', 'сокольники'],
            'hobby': ['фотографией', 'увлекается'],
            'pets': ['собака', 'рекс'],
            'education': ['изучает', 'нейросети']
        }
    
    def classify_question(self, question: str) -> str:
        question_lower = question.lower()
        for category, keywords in self.fact_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        return 'general'
    
    def extract_context_from_memory(self, memory_data: List[str], question: str) -> str:
        question_lower = question.lower()
        relevant_contexts = []
        
        for memory_item in memory_data:
            memory_lower = memory_item.lower()
            question_words = set(question_lower.split())
            memory_words = set(memory_lower.split())
            
            if question_words.intersection(memory_words):
                relevant_contexts.append(memory_item)
        
        return "\n".join(relevant_contexts[:3])
    
    def generate_enhanced_prompt(self, question: str, memory_data: List[str]) -> str:
        category = self.classify_question(question)
        context = self.extract_context_from_memory(memory_data, question)
        
        prompt_parts = [
            "Ты - помощник с доступом к структурированной памяти пользователя.",
            "Используй следующую информацию для точного ответа на вопрос:",
            "",
            "ВОПРОС: " + question,
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "КОНТЕКСТ ИЗ ПАМЯТИ:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "ИНСТРУКЦИИ:",
            "- Отвечай кратко и по существу",
            "- Используй только информацию из предоставленного контекста",
            "- Если информации недостаточно, скажи об этом",
            "- Будь точным в деталях"
        ])
        
        return "\n".join(prompt_parts)

def run_full_demo():
    """Запускает полную демонстрацию системы"""
    
    print("🎯 Полная демонстрация системы GigaMemory")
    print("=" * 60)
    
    # Инициализация системы
    system = GigaMemoryDemo()
    
    # Тестовые данные
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
    
    dialogue_id = "demo_dialogue_1"
    
    # Обработка диалога
    print(f"\n📚 Обработка диалога: {dialogue_id}")
    stats = system.process_dialogue(dialogue_id, test_messages)
    
    print(f"\n📊 Результаты обработки:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Тестовые вопросы
    test_questions = [
        "Расскажи о работе пользователя",
        "Какая у пользователя семья?",
        "Где живет пользователь?",
        "Какие у пользователя увлечения?",
        "Расскажи о домашних животных",
        "Что изучает пользователь?"
    ]
    
    # Ответы на вопросы
    print(f"\n❓ Ответы на вопросы:")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        answer = system.answer_question(dialogue_id, question)
        print(f"   Ответ: {answer}")
    
    # Финальная статистика
    print(f"\n📈 Финальная статистика системы:")
    print("=" * 60)
    
    final_stats = system.get_system_stats()
    for category, stats in final_stats.items():
        print(f"\n{category.upper()}:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n🎉 Демонстрация завершена!")
    print(f"✅ Показана полная работа системы GigaMemory")
    print(f"✅ Продемонстрирована генерация промптов")
    print(f"✅ Показана интеграция всех компонентов")

if __name__ == "__main__":
    run_full_demo()
