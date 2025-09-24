#!/usr/bin/env python
"""
Улучшенный скрипт с интеграцией FactBasedRAGEngine и FactBasedQuestionClassifier
"""
import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

# Импортируем наши модули
try:
    from extraction.fact_models import FactType, Fact, FactConfidence, FactRelation
    from questions.fact_based_classifier import FactBasedQuestionClassifier
    from rag.fact_based_rag import FactBasedRAGEngine
    from extraction.fact_database import FactDatabase
except ImportError as e:
    print(f"⚠️ Не удалось импортировать модули: {e}")
    print("Создаем упрощенную версию...")
    
    # Создаем упрощенные классы для работы
    class FactType:
        WORK_OCCUPATION = "work_occupation"
        PERSONAL_NAME = "personal_name"
        PERSONAL_AGE = "personal_age"
        FAMILY_STATUS = "family_status"
        SPORT_TYPE = "sport_type"
        PET_TYPE = "pet_type"
        TRANSPORT_CAR_BRAND = "transport_car_brand"
        DRINK_COFFEE = "drink_coffee"
        FINANCE_INCOME = "finance_income"
        EDUCATION_INSTITUTION = "education_institution"
        HOBBY_ACTIVITY = "hobby_activity"
        HEALTH_CONDITION = "health_condition"
        PROPERTY_TYPE = "property_type"
        CONTACT_PHONE = "contact_phone"
        CONTACT_EMAIL = "contact_email"
    
    class FactConfidence:
        def __init__(self, score: float):
            self.score = score
    
    class FactRelation:
        HAS = "has"
        IS = "is"
        WORKS_AS = "works_as"
        LIVES_IN = "lives_in"
        OWNS = "owns"
        LIKES = "likes"
    
    class Fact:
        def __init__(self, type: str, subject: str, relation: str, object: str, 
                     confidence: FactConfidence, session_id: str, dialogue_id: str):
            self.type = type
            self.subject = subject
            self.relation = relation
            self.object = object
            self.confidence = confidence
            self.session_id = session_id
            self.dialogue_id = dialogue_id
        
        def to_natural_text(self) -> str:
            return f"{self.subject} {self.relation} {self.object}"
    
    class FactDatabase:
        def __init__(self):
            self.facts = []
        
        def add_fact(self, fact: Fact):
            self.facts.append(fact)
        
        def query_facts(self, dialogue_id: str, fact_type: str = None, 
                       min_confidence: float = 0.0, query: str = None) -> List[Fact]:
            results = []
            for fact in self.facts:
                if fact.dialogue_id == dialogue_id:
                    if fact_type and fact.type != fact_type:
                        continue
                    if fact.confidence.score < min_confidence:
                        continue
                    if query and query.lower() not in fact.object.lower():
                        continue
                    results.append(fact)
            return results
    
    class FactBasedQuestionClassifier:
        def __init__(self):
            self.question_patterns = {
                FactType.WORK_OCCUPATION: [
                    r'кем\s+(?:я\s+)?работа',
                    r'(?:моя\s+)?(?:профессия|специальность)',
                    r'чем\s+(?:я\s+)?занима',
                ],
                FactType.PERSONAL_NAME: [
                    r'как\s+(?:меня\s+)?зовут',
                    r'(?:мое|моё)\s+имя',
                    r'кто\s+я(?:\s+такой)?',
                ],
                FactType.PERSONAL_AGE: [
                    r'сколько\s+(?:мне\s+)?лет',
                    r'(?:мой\s+)?возраст',
                    r'когда\s+(?:я\s+)?родил',
                ],
                FactType.SPORT_TYPE: [
                    r'(?:каким\s+)?спортом\s+(?:я\s+)?(?:занимаюсь|увлекаюсь)',
                    r'(?:мои\s+)?(?:тренировки|физические\s+нагрузки)',
                ],
                FactType.PET_TYPE: [
                    r'(?:какие\s+)?(?:у\s+меня\s+)?(?:питомцы|животные)',
                    r'есть\s+ли\s+(?:у\s+меня\s+)?(?:кот|кошка|собака)',
                    r'(?:какая\s+)?(?:у\s+меня\s+)?(?:собака|кошка)',
                ],
                FactType.TRANSPORT_CAR_BRAND: [
                    r'(?:какая\s+)?(?:у\s+меня\s+)?(?:машина|авто|тачка)',
                    r'(?:марка|бренд)\s+(?:моей\s+)?(?:машины|авто)',
                    r'на\s+чем\s+(?:я\s+)?(?:езжу|катаюсь)',
                ],
                FactType.DRINK_COFFEE: [
                    r'(?:какой\s+)?кофе\s+(?:я\s+)?(?:пью|люблю)',
                    r'(?:пью\s+ли\s+я\s+)?кофе',
                ],
                FactType.FINANCE_INCOME: [
                    r'(?:мой\s+)?(?:доход|заработок)',
                    r'сколько\s+(?:я\s+)?(?:получаю|имею)',
                ],
                FactType.EDUCATION_INSTITUTION: [
                    r'где\s+(?:я\s+)?(?:учился|училась|учусь)',
                    r'(?:мой\s+)?(?:университет|институт|вуз)',
                    r'(?:какое\s+)?(?:у\s+меня\s+)?образование',
                ],
                FactType.HOBBY_ACTIVITY: [
                    r'(?:мои\s+)?(?:хобби|увлечения|интересы)',
                    r'чем\s+(?:я\s+)?(?:увлекаюсь|интересуюсь)',
                    r'(?:мое|моё)\s+(?:любимое\s+)?(?:занятие|дело)',
                ],
                FactType.HEALTH_CONDITION: [
                    r'(?:мое|моё)\s+(?:здоровье|самочувствие)',
                    r'(?:чем\s+)?(?:я\s+)?(?:болею|болел)',
                    r'(?:мои\s+)?(?:болезни|заболевания)',
                ],
                FactType.PROPERTY_TYPE: [
                    r'(?:где|как)\s+(?:я\s+)?живу',
                    r'(?:моя\s+)?(?:квартира|дом|жилье)',
                    r'(?:сколько\s+)?комнат',
                ],
                FactType.CONTACT_PHONE: [
                    r'(?:мой\s+)?(?:номер|телефон)',
                    r'(?:как\s+)?(?:со\s+мной\s+)?(?:связаться|позвонить)',
                ],
                FactType.CONTACT_EMAIL: [
                    r'(?:мой\s+)?(?:email|почта|мейл)',
                    r'(?:электронная\s+)?почта',
                ],
            }
        
        def classify_question(self, question: str) -> Tuple[Optional[str], float]:
            question_lower = question.lower().strip().rstrip('?')
            
            best_match = None
            best_confidence = 0.0
            
            for fact_type, patterns in self.question_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, question_lower)
                    if match:
                        matched_length = len(match.group())
                        total_length = len(question_lower)
                        confidence = matched_length / total_length if total_length > 0 else 0.5
                        
                        if matched_length == total_length:
                            confidence = 1.0
                        
                        if confidence > best_confidence:
                            best_match = fact_type
                            best_confidence = confidence
            
            return best_match, best_confidence
    
    class FactBasedRAGEngine:
        def __init__(self, fact_database: FactDatabase):
            self.fact_database = fact_database
            self.classifier = FactBasedQuestionClassifier()
        
        def process_question(self, question: str, dialogue_id: str) -> Tuple[str, Dict]:
            fact_type, confidence = self.classifier.classify_question(question)
            
            if not fact_type:
                return self._process_unknown_question(question, dialogue_id)
            
            relevant_facts = self.fact_database.query_facts(
                dialogue_id,
                fact_type=fact_type,
                min_confidence=0.5
            )
            
            if relevant_facts:
                return self._create_fact_based_prompt(question, relevant_facts, fact_type)
            else:
                return self._create_no_info_prompt(question, fact_type)
        
        def _create_fact_based_prompt(self, question: str, facts: List[Fact], 
                                     fact_type: str) -> Tuple[str, Dict]:
            high_confidence = [f for f in facts if f.confidence.score >= 0.8]
            medium_confidence = [f for f in facts if 0.5 <= f.confidence.score < 0.8]
            
            prompt_parts = []
            
            if high_confidence:
                prompt_parts.append("ТОЧНАЯ ИНФОРМАЦИЯ:")
                for fact in high_confidence:
                    prompt_parts.append(f"• {fact.to_natural_text()}")
            
            if medium_confidence:
                prompt_parts.append("\nВОЗМОЖНАЯ ИНФОРМАЦИЯ:")
                for fact in medium_confidence:
                    prompt_parts.append(f"• {fact.to_natural_text()} (уверенность: {fact.confidence.score:.0%})")
            
            prompt_parts.append(f"\nВопрос: {question}")
            prompt_parts.append("Ответь на основе предоставленных фактов. Если информации недостаточно, так и скажи.")
            
            metadata = {
                'strategy': 'fact_based',
                'fact_type': fact_type,
                'facts_found': len(facts),
                'confidence': 0.8  # Используем фиксированное значение для упрощения
            }
            
            return "\n".join(prompt_parts), metadata
        
        def _create_no_info_prompt(self, question: str, fact_type: str) -> Tuple[str, Dict]:
            no_info_responses = {
                FactType.WORK_OCCUPATION: "В диалоге нет информации о профессии пользователя.",
                FactType.PERSONAL_NAME: "В диалоге нет информации об имени пользователя.",
                FactType.PERSONAL_AGE: "В диалоге нет информации о возрасте пользователя.",
                FactType.SPORT_TYPE: "В диалоге нет информации о спортивных занятиях пользователя.",
                FactType.PET_TYPE: "В диалоге нет информации о питомцах пользователя.",
                FactType.TRANSPORT_CAR_BRAND: "В диалоге нет информации об автомобиле пользователя.",
                FactType.DRINK_COFFEE: "В диалоге нет информации о предпочтениях в кофе.",
                FactType.FINANCE_INCOME: "В диалоге нет информации о доходах пользователя.",
                FactType.EDUCATION_INSTITUTION: "В диалоге нет информации об образовании пользователя.",
                FactType.HOBBY_ACTIVITY: "В диалоге нет информации о хобби пользователя.",
                FactType.HEALTH_CONDITION: "В диалоге нет информации о здоровье пользователя.",
                FactType.PROPERTY_TYPE: "В диалоге нет информации о жилье пользователя.",
                FactType.CONTACT_PHONE: "В диалоге нет информации о контактах пользователя.",
                FactType.CONTACT_EMAIL: "В диалоге нет информации об email пользователя.",
            }
            
            response = no_info_responses.get(
                fact_type,
                f"В диалоге нет информации для ответа на вопрос: {question}"
            )
            
            metadata = {
                'strategy': 'no_info',
                'fact_type': fact_type,
                'facts_found': 0
            }
            
            return response, metadata
        
        def _process_unknown_question(self, question: str, dialogue_id: str) -> Tuple[str, Dict]:
            all_facts = self.fact_database.query_facts(dialogue_id, query=question)
            
            if all_facts:
                prompt = f"Найдена следующая информация:\n"
                for fact in all_facts[:5]:
                    prompt += f"• {fact.to_natural_text()}\n"
                prompt += f"\nВопрос: {question}\n"
                prompt += "Ответь на основе найденной информации."
            else:
                prompt = f"В диалоге нет информации для ответа на вопрос: {question}"
            
            metadata = {
                'strategy': 'keyword_search',
                'fact_type': None,
                'facts_found': len(all_facts)
            }
            
            return prompt, metadata


def load_dialogue(file_path: str) -> Dict[str, Any]:
    """Загружает диалог из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_copy_paste_content(content: str) -> bool:
    """Проверка на копипаст"""
    content_lower = content.lower()
    
    recommendation_markers = [
        'посоветуй', 'расскажи', 'что можно', 'куда поехать',
        'чем заняться', 'побольше о', 'с каким', 'где можно',
        'а он дорогой', 'лучше взять', 'не хочу на авито',
        'какой', 'какая', 'какие', 'как', 'что', 'где', 'когда',
        'почему', 'куда', 'чем', 'расскажи', 'объясни',
        'откуда', 'откуда этот', 'расскажи принцип', 'во всех подробностях',
        'мне интересно', 'хочу узнать', 'расскажи мне',
        'что делать', 'как быть', 'что посоветуешь'
    ]
    
    for marker in recommendation_markers:
        if marker in content_lower:
            return True
    
    if len(content) > 200:
        return True
        
    question_words = ['?', 'что ', 'где ', 'как ', 'когда ', 'почему ', 
                     'куда ', 'чем ', 'какой ', 'какая ', 'какие ']
    is_question = any(q in content_lower for q in question_words)
    
    if is_question:
        return True
        
    return False


def contains_personal_info(content: str) -> bool:
    """Проверяет содержит ли сообщение личную информацию"""
    content_lower = content.lower()
    
    personal_indicators = [
        'я', 'меня', 'мой', 'моя', 'мне', 'у меня',
        'мы', 'нас', 'наш', 'наша', 'нам', 'у нас',
        'семья', 'семье', 'детей', 'жена', 'муж',
        'сын', 'дочь', 'ребенок', 'дети',
        'работаю', 'живу', 'езжу', 'имею', 'владею'
    ]
    
    return any(ind in content_lower for ind in personal_indicators)


def extract_user_messages_only(messages: List[Dict[str, Any]]) -> List[str]:
    """Извлекает ТОЛЬКО сообщения пользователя с фильтрацией"""
    user_messages = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        
        if role != "user":
            continue
            
        if not content:
            continue
            
        if is_copy_paste_content(content):
            continue
            
        if len(content) < 10 or len(content) > 300:
            continue
            
        if contains_personal_info(content):
            user_messages.append(content)
    
    return user_messages


def extract_facts_from_messages(user_messages: List[str], dialogue_id: str, 
                               session_id: str) -> List[Fact]:
    """Извлекает факты из сообщений пользователя"""
    facts = []
    
    for message in user_messages:
        message_lower = message.lower()
        
        # Извлекаем факты о работе
        if 'работаю' in message_lower:
            if 'программист' in message_lower:
                facts.append(Fact(
                    type=FactType.WORK_OCCUPATION,
                    subject="пользователь",
                    relation=FactRelation.WORKS_AS,
                    object="программист",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
            elif 'учитель' in message_lower:
                facts.append(Fact(
                    type=FactType.WORK_OCCUPATION,
                    subject="пользователь",
                    relation=FactRelation.WORKS_AS,
                    object="учитель",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
            elif 'врач' in message_lower:
                facts.append(Fact(
                    type=FactType.WORK_OCCUPATION,
                    subject="пользователь",
                    relation=FactRelation.WORKS_AS,
                    object="врач",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
        
        # Извлекаем факты о спорте
        if 'спорт' in message_lower:
            if 'костюм' in message_lower and 'отказываюсь' in message_lower:
                facts.append(Fact(
                    type=FactType.SPORT_TYPE,
                    subject="пользователь",
                    relation=FactRelation.HAS,
                    object="не носит спортивные костюмы",
                    confidence=FactConfidence(0.8),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
            elif 'футбол' in message_lower:
                facts.append(Fact(
                    type=FactType.SPORT_TYPE,
                    subject="пользователь",
                    relation=FactRelation.LIKES,
                    object="футбол",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
        
        # Извлекаем факты о питомцах
        if 'собака' in message_lower or 'пес' in message_lower:
            if 'лабрадор' in message_lower:
                facts.append(Fact(
                    type=FactType.PET_TYPE,
                    subject="пользователь",
                    relation=FactRelation.OWNS,
                    object="собака породы лабрадор",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
            elif 'овчарка' in message_lower:
                facts.append(Fact(
                    type=FactType.PET_TYPE,
                    subject="пользователь",
                    relation=FactRelation.OWNS,
                    object="собака породы овчарка",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
        
        # Извлекаем факты о транспорте
        if 'машина' in message_lower or 'авто' in message_lower:
            if 'тойота' in message_lower:
                facts.append(Fact(
                    type=FactType.TRANSPORT_CAR_BRAND,
                    subject="пользователь",
                    relation=FactRelation.OWNS,
                    object="Toyota",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
            elif 'мерседес' in message_lower:
                facts.append(Fact(
                    type=FactType.TRANSPORT_CAR_BRAND,
                    subject="пользователь",
                    relation=FactRelation.OWNS,
                    object="Mercedes",
                    confidence=FactConfidence(0.9),
                    session_id=session_id,
                    dialogue_id=dialogue_id
                ))
        
        # Извлекаем факты о возрасте
        age_match = re.search(r'(\d+)\s*лет', message_lower)
        if age_match:
            facts.append(Fact(
                type=FactType.PERSONAL_AGE,
                subject="пользователь",
                relation=FactRelation.IS,
                object=f"{age_match.group(1)} лет",
                confidence=FactConfidence(0.9),
                session_id=session_id,
                dialogue_id=dialogue_id
            ))
        
        # Извлекаем факты об имени
        name_match = re.search(r'(?:зовут|имя)\s+([а-яё]+)', message_lower)
        if name_match:
            facts.append(Fact(
                type=FactType.PERSONAL_NAME,
                subject="пользователь",
                relation=FactRelation.IS,
                object=name_match.group(1),
                confidence=FactConfidence(0.9),
                session_id=session_id,
                dialogue_id=dialogue_id
            ))
    
    return facts


def create_enhanced_prompt_from_dialogue(dialogue: Dict[str, Any], 
                                       fact_database: FactDatabase) -> str:
    """Создает улучшенный промпт на основе диалога и фактов"""
    dialogue_id = dialogue.get("id", "unknown")
    question = dialogue.get("question", "Как меня зовут?")
    
    # Собираем сообщения пользователя
    all_user_messages = []
    for session in dialogue.get("sessions", []):
        session_messages = session.get("messages", [])
        user_messages = extract_user_messages_only(session_messages)
        all_user_messages.extend(user_messages)
    
    # Извлекаем факты
    all_facts = []
    for session in dialogue.get("sessions", []):
        session_id = session.get("id", "unknown")
        session_messages = session.get("messages", [])
        user_messages = extract_user_messages_only(session_messages)
        facts = extract_facts_from_messages(user_messages, dialogue_id, session_id)
        all_facts.extend(facts)
        fact_database.facts.extend(facts)
    
    # Используем RAG движок
    rag_engine = FactBasedRAGEngine(fact_database)
    prompt, metadata = rag_engine.process_question(question, dialogue_id)
    
    # Добавляем контекст пользователя
    context_parts = [f"Диалог ID: {dialogue_id}", f"Вопрос: {question}", ""]
    
    if all_user_messages:
        context_parts.append("Информация о пользователе:")
        for i, msg in enumerate(all_user_messages[-10:], 1):
            context_parts.append(f"{i}. {msg}")
    else:
        context_parts.append("Личная информация не найдена в диалоге.")
    
    context_parts.extend(["", prompt])
    
    return "\n".join(context_parts), metadata


def process_dialogue_enhanced(dialogue: Dict[str, Any], 
                            fact_database: FactDatabase) -> Dict[str, Any]:
    """Обрабатывает диалог с использованием факт-ориентированного подхода"""
    try:
        dialogue_id = dialogue.get("id", "unknown")
        question = dialogue.get("question", "Как меня зовут?")
        
        # Создаем улучшенный промпт
        prompt, metadata = create_enhanced_prompt_from_dialogue(dialogue, fact_database)
        
        # Подсчитываем статистику
        total_messages = 0
        user_messages_count = 0
        filtered_messages_count = 0
        
        for session in dialogue.get("sessions", []):
            session_messages = session.get("messages", [])
            total_messages += len(session_messages)
            
            for msg in session_messages:
                if msg.get("role") == "user":
                    user_messages_count += 1
                    content = msg.get("content", "").strip()
                    
                    if (len(content) >= 10 and len(content) <= 300 and 
                        not is_copy_paste_content(content) and 
                        contains_personal_info(content)):
                        filtered_messages_count += 1
        
        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "prompt": prompt,
            "metadata": metadata,
            "total_messages": total_messages,
            "user_messages": user_messages_count,
            "filtered_messages": filtered_messages_count,
            "sessions_count": len(dialogue.get("sessions", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Ошибка при обработке диалога: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dialogue_id": dialogue.get("id", "unknown"),
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def run_enhanced_inference(dataset_path: str, output_path: str):
    """Запускает улучшенный инференс с факт-ориентированным подходом"""
    print(f"🚀 Запуск УЛУЧШЕННОГО инференса с факт-ориентированным подходом: {dataset_path}")
    
    # Создаем выходную директорию
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Создаем базу фактов
    fact_database = FactDatabase()
    
    # Загружаем датасет
    dialogues = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dialogues.append(json.loads(line))
    
    print(f"📖 Загружено {len(dialogues)} диалогов")
    
    # Обрабатываем каждый диалог
    results = []
    prompts = []
    
    for i, dialogue in enumerate(dialogues):
        print(f"⚙️ Обрабатываем диалог {i+1}/{len(dialogues)}: {dialogue.get('id', 'unknown')}")
        result = process_dialogue_enhanced(dialogue, fact_database)
        results.append(result)
        
        # Сохраняем промпт отдельно
        if "prompt" in result:
            prompts.append({
                "dialogue_id": result["dialogue_id"],
                "prompt": result["prompt"],
                "metadata": result.get("metadata", {})
            })
    
    # Сохраняем результаты
    output_file = output_dir / "results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Сохраняем промпты отдельно
    prompts_file = output_dir / "prompts.jsonl"
    with open(prompts_file, 'w', encoding='utf-8') as f:
        for prompt_data in prompts:
            f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
    
    # Сохраняем промпты в отдельные файлы
    prompts_dir = output_dir / "prompt_files"
    prompts_dir.mkdir(exist_ok=True)
    
    for i, prompt_data in enumerate(prompts):
        prompt_file = prompts_dir / f"prompt_{prompt_data['dialogue_id']}.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_data["prompt"])
    
    # Сохраняем базу фактов
    facts_file = output_dir / "facts.jsonl"
    with open(facts_file, 'w', encoding='utf-8') as f:
        for fact in fact_database.facts:
            fact_data = {
                "type": fact.type,
                "subject": fact.subject,
                "relation": fact.relation,
                "object": fact.object,
                "confidence": fact.confidence.score,
                "session_id": fact.session_id,
                "dialogue_id": fact.dialogue_id
            }
            f.write(json.dumps(fact_data, ensure_ascii=False) + '\n')
    
    print(f"💾 Результаты сохранены в {output_file}")
    print(f"💾 Промпты сохранены в {prompts_file}")
    print(f"💾 Отдельные файлы промптов сохранены в {prompts_dir}")
    print(f"💾 База фактов сохранена в {facts_file}")
    print(f"📊 Всего извлечено фактов: {len(fact_database.facts)}")
    print("✅ УЛУЧШЕННЫЙ инференс завершен успешно!")
    
    return 0


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Enhanced Fact-Based Dialogue Inference")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к датасету для инференса")
    parser.add_argument("--output", type=str, default="./enhanced_output", help="Путь для сохранения результатов")
    
    args = parser.parse_args()
    
    return run_enhanced_inference(args.dataset, args.output)


if __name__ == "__main__":
    sys.exit(main())
