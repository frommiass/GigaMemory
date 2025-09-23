"""
Тестовый скрипт для проверки системы GigaMemory на больших диалогах
Генерирует диалог в 100к символов и тестирует все компоненты
"""
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict
from models import Message

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DialogueGenerator:
    """Генератор тестовых диалогов с личной информацией и копипастом"""
    
    def __init__(self):
        # Шаблоны личной информации
        self.personal_templates = [
            "Меня зовут {}.",
            "Мне {} лет.",
            "Я живу в городе {}.",
            "Я работаю {}.",
            "У меня есть {} по имени {}.",
            "Моя любимая еда - {}.",
            "Я занимаюсь спортом: {}.",
            "У меня машина {}.",
            "Моя жена зовут {}.",
            "У меня {} детей.",
            "Я учился в {}.",
            "Мой любимый цвет {}.",
            "Я родился в {} году.",
            "Мой телефон {}.",
            "Я зарабатываю {} рублей.",
        ]
        
        # Шаблоны копипаста (будут отфильтрованы)
        self.copypaste_templates = [
            "Переведи этот текст: {}",
            "Вот статья про {}: [длинный текст на 1000 символов]",
            "Исправь ошибки в коде: {}",
            "Напиши сочинение на тему {}",
            "Объясни, что такое {}",
            "Сделай презентацию про {}",
            "Реши задачу: {}",
            "Проанализируй этот документ: {}",
        ]
        
        # Данные для генерации
        self.names = ["Александр", "Мария", "Иван", "Елена", "Дмитрий", "Анна"]
        self.ages = [25, 30, 35, 40, 45, 28, 33, 37]
        self.cities = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань"]
        self.professions = ["программист", "менеджер", "врач", "учитель", "инженер", "дизайнер"]
        self.pets = ["кот", "собака", "попугай", "хомяк"]
        self.pet_names = ["Мурка", "Барсик", "Шарик", "Рекс", "Кеша"]
        self.foods = ["пицца", "суши", "паста", "борщ", "пельмени", "шашлык"]
        self.sports = ["футбол", "плавание", "бег", "йога", "теннис", "бокс"]
        self.cars = ["Toyota Camry", "BMW X5", "Mercedes E-class", "Hyundai Solaris", "Kia Rio"]
        self.universities = ["МГУ", "МГТУ", "ВШЭ", "СПбГУ", "МИФИ"]
        self.colors = ["синий", "зеленый", "красный", "черный", "белый"]

    def generate_personal_message(self) -> str:
        """Генерирует сообщение с личной информацией"""
        template = random.choice(self.personal_templates)
        
        if "зовут {}" in template:
            return template.format(random.choice(self.names))
        elif "{} лет" in template:
            return template.format(random.choice(self.ages))
        elif "городе {}" in template:
            return template.format(random.choice(self.cities))
        elif "работаю {}" in template:
            return template.format(random.choice(self.professions))
        elif "{} по имени {}" in template:
            return template.format(random.choice(self.pets), random.choice(self.pet_names))
        elif "еда - {}" in template:
            return template.format(random.choice(self.foods))
        elif "спортом: {}" in template:
            return template.format(random.choice(self.sports))
        elif "машина {}" in template:
            return template.format(random.choice(self.cars))
        elif "жену зовут {}" in template:
            return template.format(random.choice(self.names))
        elif "{} детей" in template:
            return template.format(random.randint(1, 3))
        elif "учился в {}" in template:
            return template.format(random.choice(self.universities))
        elif "цвет {}" in template:
            return template.format(random.choice(self.colors))
        elif "{} году" in template:
            return template.format(random.randint(1980, 2000))
        elif "телефон {}" in template:
            return template.format(f"+7{random.randint(9000000000, 9999999999)}")
        elif "{} рублей" in template:
            return template.format(random.randint(50000, 200000))
        else:
            return "Я люблю читать книги."

    def generate_copypaste_message(self) -> str:
        """Генерирует копипаст сообщение (будет отфильтровано)"""
        template = random.choice(self.copypaste_templates)
        topic = random.choice(["Python", "машинное обучение", "нейросети", "блокчейн"])
        
        if "[длинный текст" in template:
            # Генерируем длинный копипаст
            long_text = " ".join([
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 20
            ])
            return template.format(topic).replace("[длинный текст на 1000 символов]", long_text)
        else:
            return template.format(topic)

    def generate_dialogue(self, target_chars: int = 100000) -> List[Dict]:
        """
        Генерирует диалог заданной длины
        
        Args:
            target_chars: Целевое количество символов
            
        Returns:
            Список сообщений в формате для системы
        """
        messages = []
        current_chars = 0
        session_id = 1
        messages_in_session = 0
        
        while current_chars < target_chars:
            # 70% личная информация, 30% копипаст (для реалистичности)
            if random.random() < 0.7:
                content = self.generate_personal_message()
            else:
                content = self.generate_copypaste_message()
            
            # Создаем сообщение пользователя
            messages.append({
                "role": "user",
                "content": content,
                "session_id": f"session_{session_id}"
            })
            
            current_chars += len(content)
            messages_in_session += 1
            
            # Добавляем ответ ассистента (короткий)
            assistant_response = random.choice([
                "Понял, записал.",
                "Интересно!",
                "Спасибо за информацию.",
                "Отлично!",
                "Хорошо.",
            ])
            
            messages.append({
                "role": "assistant",
                "content": assistant_response,
                "session_id": f"session_{session_id}"
            })
            
            current_chars += len(assistant_response)
            
            # Новая сессия каждые 10-20 сообщений
            if messages_in_session > random.randint(10, 20):
                session_id += 1
                messages_in_session = 0
        
        logger.info(f"Сгенерирован диалог: {len(messages)} сообщений, {current_chars} символов, {session_id} сессий")
        return messages


def test_large_dialogue(model_path: str = "/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16"):
    """
    Основной тест системы на большом диалоге
    
    Args:
        model_path: Путь к модели GigaChat
    """
    print("\n" + "="*80)
    print("🚀 ТЕСТ GIGAMEMORY НА БОЛЬШОМ ДИАЛОГЕ (100К СИМВОЛОВ)")
    print("="*80)
    
    # Импортируем систему
    from submit.model_inference import SubmitModelWithMemory
    
    # 1. Инициализация системы
    print("\n1️⃣ Инициализация системы...")
    start_time = time.time()
    
    model = SubmitModelWithMemory(model_path)
    model.optimize_for_large_dialogue()  # Оптимизация для больших диалогов
    
    init_time = time.time() - start_time
    print(f"✅ Система инициализирована за {init_time:.2f} сек")
    
    # 2. Генерация тестового диалога
    print("\n2️⃣ Генерация тестового диалога...")
    generator = DialogueGenerator()
    dialogue_messages = generator.generate_dialogue(target_chars=100000)
    
    # Конвертируем в формат Message
    messages = []
    for msg_dict in dialogue_messages:
        msg = Message(
            role=msg_dict["role"],
            content=msg_dict["content"],
            session_id=msg_dict.get("session_id")
        )
        messages.append(msg)
    
    dialogue_id = "test_dialogue_100k"
    
    # 3. Запись в память
    print("\n3️⃣ Обработка и запись в память...")
    start_time = time.time()
    
    # Обрабатываем батчами для эффективности
    batch_size = 100
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        model.write_to_memory(batch, dialogue_id)
        
        if i % 1000 == 0:
            print(f"   Обработано {i}/{len(messages)} сообщений...")
    
    write_time = time.time() - start_time
    print(f"✅ Диалог записан в память за {write_time:.2f} сек")
    
    # 4. Получаем статистику обработки
    stats = model.get_stats()
    print("\n📊 Статистика обработки:")
    print(f"   - Всего сообщений: {stats['processing_stats']['total_messages']}")
    print(f"   - Отфильтровано (личная инфо): {stats['processing_stats']['filtered_messages']}")
    print(f"   - Копипаст отфильтрован: {stats['processing_stats']['copypaste_filtered']}")
    print(f"   - Создано сессий: {stats['processing_stats']['sessions_created']}")
    print(f"   - Извлечено фактов: {stats['processing_stats']['facts_extracted']}")
    print(f"   - Коэффициент сжатия: {stats['processing_stats']['compression_ratio']:.2f}")
    print(f"   - Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
    
    # 5. Тестовые вопросы
    print("\n4️⃣ Тестирование ответов на вопросы...")
    
    test_questions = [
        "Как меня зовут?",
        "Сколько мне лет?",
        "Где я живу?",
        "Кем я работаю?",
        "Какие у меня есть питомцы?",
        "Какая у меня машина?",
        "Есть ли у меня дети?",
        "Где я учился?",
        "Какой мой любимый цвет?",
        "Какая моя любимая еда?",
        "Каким спортом я занимаюсь?",
        "Сколько я зарабатываю?",
    ]
    
    print("\n🔍 Ответы на вопросы:\n")
    for question in test_questions:
        start_time = time.time()
        answer = model.answer_to_question(dialogue_id, question)
        answer_time = time.time() - start_time
        
        print(f"❓ {question}")
        print(f"💬 {answer}")
        print(f"⏱️ Время ответа: {answer_time:.3f} сек")
        print()
    
    # 6. Итоговая статистика
    print("\n" + "="*80)
    print("📈 ИТОГОВАЯ СТАТИСТИКА")
    print("="*80)
    
    final_stats = model.get_stats()
    
    print("\n📌 Обработка:")
    print(f"   - Эффективность фильтрации: {final_stats['processing_stats']['filtered_messages']/max(1, final_stats['processing_stats']['total_messages']):.2%}")
    print(f"   - Средний размер сессии: {final_stats['processing_stats']['filtered_messages']/max(1, final_stats['processing_stats']['sessions_created']):.1f} сообщений")
    
    print("\n📌 Производительность:")
    print(f"   - Скорость обработки: {len(messages)/write_time:.0f} сообщений/сек")
    print(f"   - Использование кэша: {final_stats['cache_stats']['hit_rate']:.2%}")
    print(f"   - Размер кэша: {final_stats['cache_stats']['cache_size']} записей")
    
    print("\n📌 Память:")
    print(f"   - Диалогов в памяти: {final_stats['memory_stats']['total_dialogues']}")
    print(f"   - Сессий в памяти: {final_stats['memory_stats']['total_sessions']}")
    
    # 7. Очистка памяти
    print("\n5️⃣ Очистка памяти...")
    model.clear_memory(dialogue_id)
    print("✅ Память очищена")
    
    print("\n" + "="*80)
    print("✅ ТЕСТ УСПЕШНО ЗАВЕРШЕН!")
    print("="*80)


def create_submission_file(dialogue_file: str = "data/test_dialogue.jsonl",
                         output_file: str = "submit.csv"):
    """
    Создает файл submit.csv для отправки на проверку
    
    Args:
        dialogue_file: Путь к файлу с диалогами
        output_file: Путь к выходному файлу
    """
    from submit.model_inference import SubmitModelWithMemory
    import csv
    
    model_path = "/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16"
    model = SubmitModelWithMemory(model_path)
    
    results = []
    
    with open(dialogue_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dialogue_id = data['id']
            question = data['question']
            
            # Обрабатываем все сессии
            all_messages = []
            for session in data['sessions']:
                for msg in session['messages']:
                    message = Message(
                        role=msg['role'],
                        content=msg['content'],
                        session_id=session['id']
                    )
                    all_messages.append(message)
            
            # Записываем в память
            model.write_to_memory(all_messages, dialogue_id)
            
            # Отвечаем на вопрос
            start_time = time.time()
            answer = model.answer_to_question(dialogue_id, question)
            answer_time = time.time() - start_time
            
            results.append({
                'id': dialogue_id,
                'answer': answer,
                'answer_time': answer_time
            })
            
            # Очищаем память после каждого диалога
            model.clear_memory(dialogue_id)
    
    # Сохраняем результаты
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'answer_time'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Файл {output_file} создан с {len(results)} ответами")


if __name__ == "__main__":
    # Запуск теста
    test_large_dialogue()
    
    # Для создания файла отправки (если есть данные)
    # create_submission_file()
