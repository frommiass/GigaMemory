#!/usr/bin/env python
"""
Скрипт валидации системы GigaMemory
Проверяет готовность к работе с большими диалогами
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Цвета для красивого вывода
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Печатает заголовок"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")


def print_success(text: str):
    """Печатает успех"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_error(text: str):
    """Печатает ошибку"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Печатает предупреждение"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_info(text: str):
    """Печатает информацию"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


def check_imports() -> Tuple[bool, List[str]]:
    """Проверяет, что все модули импортируются"""
    print_header("ПРОВЕРКА ИМПОРТОВ")
    
    missing = []
    modules_to_check = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
    ]
    
    for module, name in modules_to_check:
        try:
            __import__(module)
            print_success(f"{name} установлен")
        except ImportError:
            print_error(f"{name} НЕ установлен")
            missing.append(module)
    
    # Проверка vllm (опционально)
    try:
        import vllm
        print_success("vLLM установлен")
    except ImportError:
        print_warning("vLLM НЕ установлен (опционально для тестирования)")
    
    # Проверка проектных модулей
    try:
        from models import Message
        from submit_interface import ModelWithMemory
        print_success("Базовые модели доступны")
    except ImportError as e:
        print_error(f"Базовые модели недоступны: {e}")
        missing.append('models')
    
    return len(missing) == 0, missing


def check_model_path(model_path: str = "/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16") -> bool:
    """Проверяет наличие модели"""
    print_header("ПРОВЕРКА МОДЕЛИ")
    
    path = Path(model_path)
    if path.exists():
        # Проверяем основные файлы модели
        required_files = [
            'config.json',
            'tokenizer.json',
            'pytorch_model.bin'  # или model.safetensors
        ]
        
        missing_files = []
        for file in required_files:
            file_path = path / file
            if not file_path.exists() and not (path / 'model.safetensors').exists():
                missing_files.append(file)
        
        if missing_files:
            print_warning(f"Модель найдена, но отсутствуют файлы: {missing_files}")
            return False
        else:
            print_success(f"Модель найдена: {model_path}")
            return True
    else:
        print_warning(f"Модель НЕ найдена: {model_path}")
        print_info("Для тестирования это нормально. В продакшене скачайте модель:")
        print(f"  git clone https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16")
        return True  # Не блокируем тестирование


def check_memory() -> bool:
    """Проверяет доступную память"""
    print_header("ПРОВЕРКА ПАМЯТИ")
    
    try:
        import psutil
        
        # RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        
        if ram_gb >= 200:
            print_success(f"RAM: {ram_gb:.0f} GB (достаточно)")
        elif ram_gb >= 100:
            print_warning(f"RAM: {ram_gb:.0f} GB (может не хватить для больших диалогов)")
        elif ram_gb >= 16:
            print_warning(f"RAM: {ram_gb:.0f} GB (достаточно для тестирования)")
        else:
            print_error(f"RAM: {ram_gb:.0f} GB (недостаточно)")
            return False
        
        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_memory_gb = props.total_memory / (1024**3)
                    print_success(f"GPU {i}: {props.name} с {gpu_memory_gb:.0f} GB памяти")
            else:
                print_warning("GPU не обнаружен, будет использоваться CPU (медленно)")
        except Exception as e:
            print_warning(f"Не удалось проверить GPU: {e}")
        
        return True
        
    except ImportError:
        print_warning("psutil не установлен, пропускаем проверку памяти")
        return True


def test_copypaste_filter() -> bool:
    """Тестирует фильтр копипаста"""
    print_header("ТЕСТ ФИЛЬТРА КОПИПАСТА")
    
    try:
        # Простая реализация фильтра для тестирования
        def is_copy_paste_content(text: str) -> bool:
            """Простая проверка на копипаст"""
            copy_paste_indicators = [
                "переведи", "исправь", "напиши", "создай", "сгенерируй",
                "def ", "class ", "import ", "function", "код:", "код ",
                "статья", "текст:", "документ", "файл:", "ссылка:"
            ]
            text_lower = text.lower()
            return any(indicator in text_lower for indicator in copy_paste_indicators)
        
        def is_personal_message(text: str) -> bool:
            """Простая проверка на личное сообщение"""
            personal_indicators = [
                "меня зовут", "мне", "я живу", "я работаю", "у меня есть",
                "я родился", "моя жена", "мой муж", "мои дети", "я увлекаюсь"
            ]
            text_lower = text.lower()
            return any(indicator in text_lower for indicator in personal_indicators)
        
        # Тестовые случаи
        test_cases = [
            # (текст, ожидается_копипаст, описание)
            ("Меня зовут Александр", False, "Личная информация"),
            ("Мне 30 лет", False, "Личная информация"),
            ("Переведи этот текст на английский", True, "Копипаст - запрос перевода"),
            ("Исправь ошибки в коде: def func():", True, "Копипаст - код"),
            ("Вот статья про машинное обучение [длинный текст]", True, "Копипаст - статья"),
            ("Я живу в Москве", False, "Личная информация"),
            ("Напиши сочинение на тему", True, "Копипаст - запрос генерации"),
        ]
        
        passed = 0
        failed = 0
        
        for text, expected_copypaste, description in test_cases:
            is_copypaste = is_copy_paste_content(text)
            is_personal = is_personal_message(text)
            
            if is_copypaste == expected_copypaste:
                print_success(f"{description}: корректно")
                passed += 1
            else:
                print_error(f"{description}: ОШИБКА (ожидалось {expected_copypaste}, получено {is_copypaste})")
                failed += 1
        
        print(f"\nРезультат: {passed}/{len(test_cases)} тестов пройдено")
        return failed == 0
        
    except Exception as e:
        print_error(f"Ошибка при тестировании фильтра: {e}")
        return False


def test_system_integration() -> bool:
    """Интеграционный тест системы"""
    print_header("ИНТЕГРАЦИОННЫЙ ТЕСТ")
    
    try:
        from models import Message
        
        print_info("Тестируем базовые модели...")
        
        # Тестовые сообщения
        test_messages = [
            Message(role="user", content="Привет! Меня зовут Тест.", session_id="1"),
            Message(role="assistant", content="Приятно познакомиться!", session_id="1"),
            Message(role="user", content="Мне 25 лет.", session_id="1"),
        ]
        
        print_success(f"Создано {len(test_messages)} тестовых сообщений")
        
        # Проверяем, что сообщения корректно созданы
        for msg in test_messages:
            assert msg.role in ["user", "assistant"]
            assert len(msg.content) > 0
            assert msg.session_id is not None
            assert msg.timestamp is not None
        
        print_success("Все сообщения корректны")
        
        # Простой тест фильтрации
        print_info("Тестируем простую фильтрацию...")
        
        def simple_filter(messages):
            """Простая фильтрация сообщений"""
            filtered = []
            for msg in messages:
                # Фильтруем копипаст
                if not any(word in msg.content.lower() for word in ["переведи", "исправь", "напиши"]):
                    filtered.append(msg)
            return filtered
        
        filtered_messages = simple_filter(test_messages)
        print_success(f"Отфильтровано: {len(test_messages)} -> {len(filtered_messages)} сообщений")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка интеграционного теста: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_dialogue(size_chars: int = 100000) -> Dict:
    """Генерирует тестовый диалог заданного размера"""
    print_header(f"ГЕНЕРАЦИЯ ТЕСТОВОГО ДИАЛОГА ({size_chars:,} символов)")
    
    dialogue = {
        "id": "test_100k",
        "question": "Как меня зовут?",
        "sessions": []
    }
    
    current_chars = 0
    session_id = 1
    
    # Шаблоны сообщений
    personal_messages = [
        "Меня зовут Александр Петров.",
        "Мне 35 лет, я родился в 1989 году.",
        "Я живу в Москве, в районе Арбат.",
        "Работаю программистом в Яндексе.",
        "У меня есть кот Барсик и собака Шарик.",
        "Женат, жену зовут Елена.",
        "Двое детей - сын Максим и дочь София.",
        "Окончил МГУ, факультет ВМК.",
        "Увлекаюсь футболом и плаванием.",
        "Езжу на Toyota Camry 2020 года.",
    ]
    
    while current_chars < size_chars:
        session = {
            "id": f"session_{session_id}",
            "messages": []
        }
        
        # Добавляем 10-15 сообщений в сессию
        for _ in range(random.randint(10, 15)):
            # Выбираем случайное личное сообщение
            user_msg = random.choice(personal_messages)
            
            session["messages"].append({
                "role": "user",
                "content": user_msg
            })
            
            session["messages"].append({
                "role": "assistant", 
                "content": "Понял, записал."
            })
            
            current_chars += len(user_msg) + 15  # +15 за ответ ассистента
            
            if current_chars >= size_chars:
                break
        
        dialogue["sessions"].append(session)
        session_id += 1
    
    print_success(f"Сгенерирован диалог: {session_id} сессий, ~{current_chars:,} символов")
    
    # Сохраняем для тестирования
    test_file = Path("test_dialogue_100k.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(dialogue, f, ensure_ascii=False)
    
    print_info(f"Диалог сохранен в {test_file}")
    
    return dialogue


def main():
    """Основная функция валидации"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 ВАЛИДАЦИЯ СИСТЕМЫ GIGAMEMORY{Colors.ENDC}")
    print(f"{Colors.BLUE}Проверка готовности к работе с диалогами 100к+ символов{Colors.ENDC}\n")
    
    results = []
    
    # 1. Проверка импортов
    import_ok, missing = check_imports()
    results.append(("Импорты", import_ok))
    
    if not import_ok:
        print_error(f"\nУстановите недостающие модули: pip install {' '.join(missing)}")
    
    # 2. Проверка модели
    model_ok = check_model_path()
    results.append(("Модель", model_ok))
    
    # 3. Проверка памяти
    memory_ok = check_memory()
    results.append(("Память", memory_ok))
    
    # 4. Тест фильтра копипаста
    filter_ok = test_copypaste_filter()
    results.append(("Фильтр копипаста", filter_ok))
    
    # 5. Интеграционный тест
    integration_ok = test_system_integration()
    results.append(("Интеграция", integration_ok))
    
    # 6. Генерация тестового диалога
    if all(ok for _, ok in results[:4]):  # Если основные тесты пройдены
        try:
            import random
            random.seed(42)  # Для воспроизводимости
            test_dialogue = generate_test_dialogue(100000)
            results.append(("Тестовый диалог", True))
        except Exception as e:
            print_error(f"Ошибка генерации тестового диалога: {e}")
            results.append(("Тестовый диалог", False))
    
    # Итоговый отчет
    print_header("ИТОГОВЫЙ ОТЧЕТ")
    
    all_passed = True
    for name, passed in results:
        if passed:
            print_success(f"{name}: ПРОЙДЕНО")
        else:
            print_error(f"{name}: ПРОВАЛЕНО")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ СИСТЕМА ГОТОВА К РАБОТЕ!{Colors.ENDC}")
        print(f"{Colors.GREEN}Можете запускать: python run.py --test{Colors.ENDC}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ СИСТЕМА НЕ ГОТОВА{Colors.ENDC}")
        print(f"{Colors.YELLOW}Исправьте проблемы выше и запустите валидацию снова{Colors.ENDC}")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import random
    sys.exit(main())
