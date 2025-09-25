"""
Паттерны для извлечения фактов из текста - УЛУЧШЕННАЯ ВЕРСИЯ
"""
import re
from typing import Dict, List, Tuple, Optional, Pattern
from .fact_models import FactType, FactRelation


# Расширенные паттерны для извлечения фактов
FACT_PATTERNS: Dict[FactType, List[Pattern]] = {
    
    # === ЛИЧНАЯ ИНФОРМАЦИЯ - УЛУЧШЕННЫЕ ПАТТЕРНЫ ===
    FactType.PERSONAL_NAME: [
        re.compile(r'(?:меня зовут|я\s*[-–—]\s*|мое имя\s*[-–—]?\s*)([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:зовут|называют)\s+(?:меня\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:я|мое имя)\s+([А-ЯЁ][а-яё]+)(?:\s|,|\.)', re.IGNORECASE),
        re.compile(r'(?:это|привет,?\s*я)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'^([А-ЯЁ][а-яё]+)\s+(?:здесь|тут)', re.IGNORECASE),
        re.compile(r'(?:можете называть меня|зовите меня)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:имя|фамилия)\s*[:–—]\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_NICKNAME: [
        re.compile(r'(?:ник|никнейм|псевдоним)\s*[:–—]?\s*([A-Za-zА-Яа-яё0-9_]+)', re.IGNORECASE),
        re.compile(r'(?:называют|кличут)\s+(?:меня\s+)?([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:прозвище|кличка)\s*[:–—]?\s*([А-Яа-яё]+)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_AGE: [
        re.compile(r'(?:мне|мой возраст)\s*[-–—]?\s*(\d+)\s*(?:лет|год)', re.IGNORECASE),
        re.compile(r'(\d+)\s*(?:лет|год)(?:а|ов)?\s*(?:мне|исполнилось)', re.IGNORECASE),
        re.compile(r'я\s+(\d+)[-\s]?летн', re.IGNORECASE),
        re.compile(r'возраст\s*[-–—:]\s*(\d+)', re.IGNORECASE),
        re.compile(r'(?:мне уже|мне всего|мне только)\s*(\d+)', re.IGNORECASE),
        re.compile(r'(?:родился|родилась)\s+в\s+(\d{4})', re.IGNORECASE),
        re.compile(r'(\d+)\s*(?:годик|годиков)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_LOCATION: [
        re.compile(r'(?:живу|проживаю|нахожусь|обитаю)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+(?:[-\s]+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:из|родом из|приехал из)\s+([А-ЯЁ][а-яё]+(?:[-\s]+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:мой город|место жительства|резиденция)\s*[-–—:]\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:переехал|переезжаю)\s+в\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:квартира|дом)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:прописан|зарегистрирован)\s+в\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_HOMETOWN: [
        re.compile(r'(?:родной город|родина|родом из)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:родился|родилась|вырос|выросла)\s+в\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:мой родной)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.PERSONAL_DISTRICT: [
        re.compile(r'(?:район|микрорайон)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'живу\s+в\s+([А-ЯЁ][а-яё]+)\s+районе', re.IGNORECASE),
        re.compile(r'(?:метро|станция)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    # === РАБОТА - УЛУЧШЕННЫЕ ПАТТЕРНЫ ===
    FactType.WORK_OCCUPATION: [
        re.compile(r'(?:работаю|тружусь|я)\s+([а-яё]+(?:ом|ером|ором|истом|ником|щиком|телем|ентом))', re.IGNORECASE),
        re.compile(r'(?:моя профессия|профессия|должность|специальность)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'я\s+(?:по профессии\s+)?([а-яё]+(?:ер|ор|ист|ник|щик|тель|ент|лог|граф))', re.IGNORECASE),
        re.compile(r'(?:занимаюсь|специализируюсь на)\s+([а-яё]+(?:ией|кой|ством))', re.IGNORECASE),
        re.compile(r'(?:по образованию|по специальности)\s+(?:я\s+)?([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.WORK_COMPANY: [
        re.compile(r'работаю\s+(?:в|на)\s+(?:компании\s+)?[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:компания|фирма|организация|корпорация)\s*[-–—:]\s*[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:в|на)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)\s+работаю', re.IGNORECASE),
        re.compile(r'(?:сотрудник|работник)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:офис в|офис)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.WORK_SALARY: [
        re.compile(r'(?:зарплата|оклад|доход)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)\s*(?:руб|долл|евро|тыс)?', re.IGNORECASE),
        re.compile(r'(?:получаю|зарабатываю)\s+(\d+(?:\s?\d{3})*)\s*(?:руб|долл|евро|тыс)?', re.IGNORECASE),
        re.compile(r'(\d+(?:\s?\d{3})*)\s*(?:руб|долл|евро|тыс)?\s+(?:в месяц|в год)', re.IGNORECASE),
    ],
    
    # === ТРАНСПОРТ И АВТОМОБИЛИ - НОВЫЕ ПАТТЕРНЫ ===
    FactType.TRANSPORT_CAR_BRAND: [
        re.compile(r'(?:машина|авто|автомобиль|тачка)\s*[:–—]?\s*([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:вожу|езжу на|купил)\s+([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:марка)\s*[:–—]?\s*([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'у меня\s+([A-Za-zА-Яа-яё]+(?:\s+[A-Za-zА-Яа-яё0-9]+)?)', re.IGNORECASE),
    ],
    
    FactType.TRANSPORT_CAR_MODEL: [
        re.compile(r'(?:модель)\s*[:–—]?\s*([A-Za-z0-9А-Яа-яё\s]+)', re.IGNORECASE),
        re.compile(r'([A-Za-zА-Яа-яё]+)\s+([A-Za-z0-9]+)\s+(?:года|г\.)', re.IGNORECASE),
    ],
    
    FactType.TRANSPORT_CAR_COLOR: [
        re.compile(r'(?:цвет машины|машина)\s+([а-яё]+(?:ая|ой|ый))', re.IGNORECASE),
        re.compile(r'([а-яё]+(?:ая|ой|ый))\s+(?:машина|авто)', re.IGNORECASE),
    ],
    
    FactType.TRANSPORT_DRIVING_LICENSE: [
        re.compile(r'(?:права|водительские права)\s+категории\s+([A-Z]+)', re.IGNORECASE),
        re.compile(r'(?:вожу|умею водить)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:стаж вождения)\s*[:–—]?\s*(\d+)\s*(?:лет|год)', re.IGNORECASE),
    ],
    
    # === ФИНАНСЫ И ДЕНЬГИ - НОВЫЕ ПАТТЕРНЫ ===
    FactType.FINANCE_INCOME: [
        re.compile(r'(?:доход|зарабатываю|получаю)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)\s*(?:руб|долл|евро|тыс)?', re.IGNORECASE),
        re.compile(r'(?:заработок|прибыль)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)', re.IGNORECASE),
    ],
    
    FactType.FINANCE_SAVINGS: [
        re.compile(r'(?:накопления|сбережения|отложил)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)\s*(?:руб|долл|евро|тыс)?', re.IGNORECASE),
        re.compile(r'(?:на счету|в банке)\s+(\d+(?:\s?\d{3})*)', re.IGNORECASE),
        re.compile(r'(?:коплю на|откладываю на)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.FINANCE_INVESTMENT: [
        re.compile(r'(?:инвестирую|вкладываю)\s+(?:в\s+)?([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:акции|облигации|фонд)\s+([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:портфель|инвестиции)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)', re.IGNORECASE),
    ],
    
    FactType.FINANCE_CRYPTO: [
        re.compile(r'(?:крипта|криптовалюта|биткоин|эфир)\s*[:–—]?\s*([A-Za-z]+)', re.IGNORECASE),
        re.compile(r'(?:держу|храню)\s+(\d+(?:\.\d+)?)\s*(?:BTC|ETH|USDT)', re.IGNORECASE),
        re.compile(r'(?:майню|торгую)\s+([A-Za-z]+)', re.IGNORECASE),
    ],
    
    FactType.FINANCE_BANK: [
        re.compile(r'(?:банк|обслуживаюсь в)\s*[:–—]?\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:карта|карточка|счет в)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:Сбер|Тинькофф|ВТБ|Альфа|Райффайзен)', re.IGNORECASE),
    ],
    
    FactType.FINANCE_CREDIT: [
        re.compile(r'(?:кредит|займ|долг)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)\s*(?:руб|долл|евро)?', re.IGNORECASE),
        re.compile(r'(?:взял|оформил)\s+кредит\s+(?:на\s+)?(\d+(?:\s?\d{3})*)', re.IGNORECASE),
        re.compile(r'(?:плачу|выплачиваю)\s+(\d+(?:\s?\d{3})*)\s*(?:руб|в месяц)?', re.IGNORECASE),
    ],
    
    FactType.FINANCE_MORTGAGE: [
        re.compile(r'(?:ипотека)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)\s*(?:руб|долл)?', re.IGNORECASE),
        re.compile(r'(?:ипотечный платеж)\s*[:–—]?\s*(\d+(?:\s?\d{3})*)', re.IGNORECASE),
        re.compile(r'(?:квартира в ипотеку|ипотека на)\s+(\d+)\s*(?:лет|год)', re.IGNORECASE),
    ],
    
    # === ЕДА И НАПИТКИ - УЛУЧШЕННЫЕ И НОВЫЕ ПАТТЕРНЫ ===
    FactType.FOOD_FAVORITE: [
        re.compile(r'(?:люблю|обожаю|предпочитаю)\s+(?:есть|кушать)?\s*([а-яё]+(?:у|ы|и|ью)?)', re.IGNORECASE),
        re.compile(r'(?:любимая еда|любимое блюдо)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:фанат|фанатею от)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:не могу без)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.FOOD_CUISINE: [
        re.compile(r'(?:кухня|кухню)\s*[:–—]?\s*([а-яё]+(?:ая|ская))', re.IGNORECASE),
        re.compile(r'(?:люблю)\s+([а-яё]+(?:скую|ую))\s+кухню', re.IGNORECASE),
        re.compile(r'(?:предпочитаю)\s+([а-яё]+(?:скую|ую))\s+еду', re.IGNORECASE),
    ],
    
    FactType.FOOD_RESTAURANT: [
        re.compile(r'(?:ресторан|ресторане)\s*[:–—]?\s*[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:хожу|ем|обедаю)\s+в\s+[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:любимый ресторан)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.FOOD_CAFE: [
        re.compile(r'(?:кафе|кофейня|кафешка)\s*[:–—]?\s*[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:пью кофе в|встречаюсь в)\s+[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:Старбакс|Starbucks|Шоколадница|Кофемания)', re.IGNORECASE),
    ],
    
    FactType.FOOD_DELIVERY: [
        re.compile(r'(?:заказываю|доставка из)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:Яндекс\.Еда|Delivery Club|Деливери)', re.IGNORECASE),
    ],
    
    # === НАПИТКИ - НОВЫЕ ПАТТЕРНЫ ===
    FactType.DRINK_COFFEE: [
        re.compile(r'(?:пью|люблю)\s+([а-яё]+)\s+кофе', re.IGNORECASE),
        re.compile(r'(?:кофе)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:латте|капучино|эспрессо|американо|раф)', re.IGNORECASE),
        re.compile(r'(\d+)\s+(?:чашек|чашки|чашка)\s+кофе', re.IGNORECASE),
    ],
    
    FactType.DRINK_TEA: [
        re.compile(r'(?:пью|люблю)\s+([а-яё]+)\s+чай', re.IGNORECASE),
        re.compile(r'(?:чай)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:зеленый|черный|белый|пуэр|улун)\s+чай', re.IGNORECASE),
    ],
    
    FactType.DRINK_ALCOHOL: [
        re.compile(r'(?:пью|выпиваю|люблю)\s+([а-яё]+(?:ое|ий|ую)?)\s*(?:вино|пиво|виски|водку)?', re.IGNORECASE),
        re.compile(r'(?:алкоголь|спиртное)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:не пью|не употребляю)\s+(?:алкоголь|спиртное)', re.IGNORECASE),
    ],
    
    FactType.DRINK_BEER: [
        re.compile(r'(?:пиво|пивко)\s*[:–—]?\s*([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:люблю|предпочитаю)\s+([а-яё]+)\s+пиво', re.IGNORECASE),
        re.compile(r'(?:светлое|темное|нефильтрованное)\s+пиво', re.IGNORECASE),
    ],
    
    FactType.DRINK_WINE: [
        re.compile(r'(?:вино)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:красное|белое|розовое|сухое|полусладкое)\s+вино', re.IGNORECASE),
        re.compile(r'(?:люблю|предпочитаю)\s+([а-яё]+)\s+вино', re.IGNORECASE),
    ],
    
    # === ПУТЕШЕСТВИЯ - РАСШИРЕННЫЕ ПАТТЕРНЫ ===
    FactType.TRAVEL_COUNTRY: [
        re.compile(r'(?:был|была|ездил|летал|путешествовал)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ]?[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:посетил|посетила)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:страна|страны)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:отдыхал|отдыхала)\s+(?:в|на)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.TRAVEL_CITY: [
        re.compile(r'(?:город|города)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:был в городе|посетил город)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.TRAVEL_DREAM: [
        re.compile(r'(?:мечтаю поехать|хочу посетить|хочу в)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:dream destination|место мечты)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.TRAVEL_FREQUENCY: [
        re.compile(r'(?:путешествую|езжу)\s+(\d+)\s+раз(?:а)?\s+в\s+(?:год|месяц)', re.IGNORECASE),
        re.compile(r'(?:раз в)\s+([а-яё]+)\s+(?:путешествую|летаю)', re.IGNORECASE),
    ],
    
    # === СПОРТ - РАСШИРЕННЫЕ ПАТТЕРНЫ ===
    FactType.SPORT_TYPE: [
        re.compile(r'(?:играю|занимаюсь)\s+(?:в\s+)?([а-яё]+(?:ом|болом|сом|кетболом|ингом|нисом))', re.IGNORECASE),
        re.compile(r'(?:тренируюсь|хожу на)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:спорт)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:бегаю|плаваю|катаюсь)', re.IGNORECASE),
        re.compile(r'(?:йога|пилатес|кроссфит|бокс|фитнес)', re.IGNORECASE),
    ],
    
    FactType.SPORT_GYM: [
        re.compile(r'(?:зал|спортзал|тренажерка|фитнес-клуб)\s*[:–—]?\s*[«"]?([^»"]+)[»"]?', re.IGNORECASE),
        re.compile(r'(?:хожу в|тренируюсь в)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:World Class|Alex Fitness|X-Fit|Физкульт)', re.IGNORECASE),
    ],
    
    FactType.SPORT_FREQUENCY: [
        re.compile(r'(?:тренируюсь|занимаюсь)\s+(\d+)\s+раз(?:а)?\s+в\s+неделю', re.IGNORECASE),
        re.compile(r'(?:каждый день|ежедневно|по утрам)\s+(?:бегаю|тренируюсь)', re.IGNORECASE),
        re.compile(r'(\d+)\s+(?:тренировок|тренировки)\s+в\s+неделю', re.IGNORECASE),
    ],
    
    # === ОБРАЗОВАНИЕ - УЛУЧШЕННЫЕ ПАТТЕРНЫ ===
    FactType.EDUCATION_INSTITUTION: [
        re.compile(r'(?:учился|училась|окончил|закончил|закончила)\s+([А-ЯЁ][А-ЯЁа-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:университет|институт|вуз|колледж|техникум|школа)\s*[-–—:]\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:в|во)\s+([А-ЯЁ][А-ЯЁа-яё]+(?:\s+[а-яё]+)*)\s+(?:учился|училась|учусь)', re.IGNORECASE),
        re.compile(r'(?:выпускник|выпускница)\s+([А-ЯЁ][А-ЯЁа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:МГУ|МГТУ|МИФИ|ВШЭ|РАНХиГС|МГИМО|СПбГУ)', re.IGNORECASE),
    ],
    
    FactType.EDUCATION_SPECIALITY: [
        re.compile(r'(?:специальность|факультет|направление)\s*[:–—]?\s*([а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:учился на|училась на|изучал|изучала)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:диплом по|степень в)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.EDUCATION_DEGREE: [
        re.compile(r'(?:бакалавр|магистр|специалист|аспирант|кандидат|доктор)', re.IGNORECASE),
        re.compile(r'(?:степень)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.EDUCATION_COURSE: [
        re.compile(r'(?:курсы|курс)\s*[:–—]?\s*([а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:прошел|прошла|окончил)\s+курсы?\s+(?:по\s+)?([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:Coursera|Udemy|Skillbox|GeekBrains|Яндекс\.Практикум)', re.IGNORECASE),
    ],
    
    # === ХОББИ - ДОПОЛНИТЕЛЬНЫЕ ПАТТЕРНЫ ===
    FactType.HOBBY_GAMING: [
        re.compile(r'(?:играю в|люблю играть в)\s+([A-Za-zА-Яа-яё0-9\s]+)', re.IGNORECASE),
        re.compile(r'(?:геймер|игры)\s*[:–—]?\s*([A-Za-zА-Яа-яё]+)', re.IGNORECASE),
        re.compile(r'(?:PlayStation|Xbox|Nintendo|Steam|PS\d)', re.IGNORECASE),
    ],
    
    FactType.HOBBY_READING: [
        re.compile(r'(?:читаю|люблю читать)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:книги|литература)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:любимый жанр)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.HOBBY_MUSIC: [
        re.compile(r'(?:слушаю|люблю)\s+([а-яё]+(?:\s+[а-яё]+)?)\s*(?:музыку)?', re.IGNORECASE),
        re.compile(r'(?:музыка|жанр)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:рок|поп|джаз|классика|рэп|электронная)', re.IGNORECASE),
    ],
    
    FactType.HOBBY_INSTRUMENT: [
        re.compile(r'(?:играю на|умею играть на)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:гитара|пианино|барабаны|скрипка|флейта)', re.IGNORECASE),
    ],
    
    # === ТЕХНОЛОГИИ - НОВЫЕ ПАТТЕРНЫ ===
    FactType.TECH_PHONE: [
        re.compile(r'(?:телефон|смартфон)\s*[:–—]?\s*([A-Za-z0-9\s]+)', re.IGNORECASE),
        re.compile(r'(?:iPhone|Samsung|Xiaomi|Huawei|OnePlus)\s*([A-Za-z0-9\s]+)?', re.IGNORECASE),
    ],
    
    FactType.TECH_LAPTOP: [
        re.compile(r'(?:ноутбук|ноут|лэптоп)\s*[:–—]?\s*([A-Za-z0-9\s]+)', re.IGNORECASE),
        re.compile(r'(?:MacBook|ThinkPad|Dell|HP|Asus|Lenovo)', re.IGNORECASE),
    ],
    
    FactType.TECH_PROGRAMMING: [
        re.compile(r'(?:программирую на|пишу на|кодирую на)\s+([A-Za-z\+\#]+)', re.IGNORECASE),
        re.compile(r'(?:язык программирования|язык)\s*[:–—]?\s*([A-Za-z\+\#]+)', re.IGNORECASE),
        re.compile(r'(?:Python|Java|JavaScript|C\+\+|C\#|Go|Rust|PHP|Ruby)', re.IGNORECASE),
    ],
    
    # === СЕМЬЯ - УЛУЧШЕННЫЕ ПАТТЕРНЫ ===
    FactType.FAMILY_SPOUSE: [
        re.compile(r'(?:жену?|мужа?|супругу?|супруга?)\s+зовут\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:женат|замужем)\s+(?:на|за)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:моя жена|мой муж|моя супруга|мой супруг)\s*[-–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:вторая половинка|половинка)\s*[:–—]?\s*([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.FAMILY_CHILDREN: [
        re.compile(r'(?:сын|сына|дочь|дочка|дочку|ребенок|ребенка)\s+(?:зовут\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:у меня\s+)?(\d+)\s+(?:детей|ребенок|ребенка|дочь|дочери|сын|сына|сыновей)', re.IGNORECASE),
        re.compile(r'(?:детей|детки|дети)\s*[-–—:]\s*(\d+)', re.IGNORECASE),
        re.compile(r'(?:старший|младший|средний)\s+(?:сын|дочь)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    # === ПИТОМЦЫ - УЛУЧШЕННЫЕ ПАТТЕРНЫ ===
    FactType.PET_NAME: [
        re.compile(r'(?:кот|кота|кошк|пес|пёс|пса|собак|питомец|питомца)\s+(?:по имени\s+|зовут\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'зовут\s+(?:моего|мою|моё)?\s*(?:кота|кошку|пса|собаку)\s+([А-ЯЁ][а-яё]+)', re.IGNORECASE),
        re.compile(r'([А-ЯЁ][а-яё]+)\s*[-–—]\s*(?:мой|моя|моё)\s+(?:кот|кошка|пес|собака)', re.IGNORECASE),
        re.compile(r'(?:попугай|попугая|хомяк|хомяка|рыбки)\s+(?:зовут\s+)?([А-ЯЁ][а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.PET_TYPE: [
        re.compile(r'(?:у меня\s+(?:есть\s+)?|завел|завела|держу|живет)\s+(кот|кошка|пес|пёс|собака|хомяк|попугай|рыбки|черепаха|кролик)', re.IGNORECASE),
        re.compile(r'(?:мой|моя|моё|мои)\s+(кот|кошка|пес|собака|хомяк|попугай|рыбки)', re.IGNORECASE),
        re.compile(r'(?:питомец)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.PET_BREED: [
        re.compile(r'(?:порода|породы)\s*[-–—:]\s*([а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'([а-яё]+(?:\s+[а-яё]+)*)\s+(?:порода|породы)', re.IGNORECASE),
        re.compile(r'(?:кот|кошка|пес|собака)\s+породы\s+([а-яё]+(?:\s+[а-яё]+)*)', re.IGNORECASE),
        re.compile(r'(?:мейн-кун|британец|сфинкс|перс|сиамский)', re.IGNORECASE),
        re.compile(r'(?:овчарка|лабрадор|хаски|корги|шпиц|бульдог|такса)', re.IGNORECASE),
    ],
    
    # === ЗДОРОВЬЕ - УЛУЧШЕННЫЕ ПАТТЕРНЫ ===
    FactType.HEALTH_CONDITION: [
        re.compile(r'(?:болею|страдаю|есть|имею|диагноз)\s+([а-яё]+(?:ией|ией|зом|тис|ия|оз))', re.IGNORECASE),
        re.compile(r'(?:диагноз|заболевание|болезнь)\s*[-–—:]\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:проблемы с|проблема с|болит|болят)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:здоров|здорова|в порядке)', re.IGNORECASE),
    ],
    
    FactType.HEALTH_ALLERGY: [
        re.compile(r'(?:аллергия на|аллергик)\s+([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:не переношу|непереносимость)\s+([а-яё]+)', re.IGNORECASE),
    ],
    
    FactType.HEALTH_MEDICATION: [
        re.compile(r'(?:принимаю|пью)\s+([а-яё]+)\s*(?:таблетки|лекарство)?', re.IGNORECASE),
        re.compile(r'(?:лекарство|препарат)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
    ],
    
    # === НЕДВИЖИМОСТЬ - НОВЫЕ ПАТТЕРНЫ ===
    FactType.PROPERTY_TYPE: [
        re.compile(r'(?:живу в|снимаю|купил)\s+([а-яё]+комнатную)\s+(?:квартиру)?', re.IGNORECASE),
        re.compile(r'(?:квартира|дом|студия|комната)\s*[:–—]?\s*([а-яё]+)', re.IGNORECASE),
        re.compile(r'(?:однушка|двушка|трешка|студия)', re.IGNORECASE),
    ],
    
    FactType.PROPERTY_AREA: [
        re.compile(r'(\d+)\s*(?:кв\.?\s*м|квадратных метров|квадратов)', re.IGNORECASE),
        re.compile(r'(?:площадь)\s*[:–—]?\s*(\d+)', re.IGNORECASE),
    ],
    
    # === КОНТАКТЫ И СОЦСЕТИ - НОВЫЕ ПАТТЕРНЫ ===
    FactType.CONTACT_PHONE: [
        re.compile(r'(?:телефон|номер|мобильный)\s*[:–—]?\s*(\+?\d[\d\s\-\(\)]+)', re.IGNORECASE),
        re.compile(r'(\+7|8)\s*[\(\s]?\d{3}[\)\s]?\s*\d{3}[\s\-]?\d{2}[\s\-]?\d{2}', re.IGNORECASE),
    ],
    
    FactType.CONTACT_EMAIL: [
        re.compile(r'(?:email|почта|мейл)\s*[:–—]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE),
        re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE),
    ],
    
    FactType.CONTACT_TELEGRAM: [
        re.compile(r'(?:telegram|телеграм|тг)\s*[:–—]?\s*@?([a-zA-Z0-9_]+)', re.IGNORECASE),
        re.compile(r'@([a-zA-Z0-9_]+)', re.IGNORECASE),
    ],
    
    FactType.CONTACT_INSTAGRAM: [
        re.compile(r'(?:instagram|инстаграм|инста)\s*[:–—]?\s*@?([a-zA-Z0-9_.]+)', re.IGNORECASE),
        re.compile(r'(?:инста)\s+([a-zA-Z0-9_.]+)', re.IGNORECASE),
    ],
}


# Паттерны для извлечения отношений
RELATION_PATTERNS: Dict[FactRelation, List[Pattern]] = {
    FactRelation.IS: [
        re.compile(r'(?:я|это|являюсь|есть)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.HAS: [
        re.compile(r'(?:у меня|имею|есть)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.WORKS_AS: [
        re.compile(r'работаю\s+(?:как\s+)?(.+)', re.IGNORECASE),
    ],
    FactRelation.WORKS_AT: [
        re.compile(r'работаю\s+(?:в|на)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.LIVES_IN: [
        re.compile(r'живу\s+(?:в|на)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.LIKES: [
        re.compile(r'(?:люблю|нравится|обожаю)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.OWNS: [
        re.compile(r'(?:владею|имею|есть)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.EARNS: [
        re.compile(r'(?:зарабатываю|получаю)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.DRINKS: [
        re.compile(r'(?:пью|выпиваю)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.DRIVES: [
        re.compile(r'(?:вожу|езжу на)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.INVESTS_IN: [
        re.compile(r'(?:инвестирую в|вкладываю в)\s+(.+)', re.IGNORECASE),
    ],
    FactRelation.TRAINS_AT: [
        re.compile(r'(?:тренируюсь в|хожу в)\s+(.+)', re.IGNORECASE),
    ],
}


# Временные маркеры
TEMPORAL_PATTERNS = {
    'past': [
        re.compile(r'(?:раньше|прежде|когда-то|в прошлом|ранее|до этого)', re.IGNORECASE),
        re.compile(r'(?:был|была|были)\s+(.+)', re.IGNORECASE),
        re.compile(r'(?:в|во?)\s+(\d{4})\s*(?:году)?', re.IGNORECASE),
        re.compile(r'(?:год назад|года назад|лет назад)', re.IGNORECASE),
    ],
    'future': [
        re.compile(r'(?:буду|будет|планирую|собираюсь|хочу|намерен)', re.IGNORECASE),
        re.compile(r'(?:в следующем|через|скоро|вскоре|завтра)', re.IGNORECASE),
        re.compile(r'(?:в планах|планирую)', re.IGNORECASE),
    ],
    'current': [
        re.compile(r'(?:сейчас|теперь|в данный момент|на данный момент|сегодня|щас)', re.IGNORECASE),
        re.compile(r'(?:в настоящее время|в настоящий момент)', re.IGNORECASE),
    ],
}


def compile_patterns() -> Dict[str, List[Pattern]]:
    """Компилирует все паттерны для оптимизации"""
    compiled = {}
    for fact_type, patterns in FACT_PATTERNS.items():
        compiled[fact_type.value] = patterns
    return compiled


def get_fact_pattern(fact_type: FactType) -> List[Pattern]:
    """Возвращает паттерны для конкретного типа факта"""
    return FACT_PATTERNS.get(fact_type, [])


def extract_with_pattern(text: str, pattern: Pattern) -> Optional[str]:
    """
    Извлекает данные по паттерну
    
    Args:
        text: Текст для поиска
        pattern: Регулярное выражение
        
    Returns:
        Извлеченное значение или None
    """
    match = pattern.search(text)
    if match and len(match.groups()) > 0:
        return match.group(1).strip()
    return None


def extract_all_with_patterns(text: str, patterns: List[Pattern]) -> List[str]:
    """
    Извлекает все совпадения по списку паттернов
    
    Args:
        text: Текст для поиска
        patterns: Список регулярных выражений
        
    Returns:
        Список извлеченных значений
    """
    results = []
    for pattern in patterns:
        value = extract_with_pattern(text, pattern)
        if value and value not in results:
            results.append(value)
    return results


def detect_temporal_context(text: str) -> Optional[str]:
    """
    Определяет временной контекст высказывания
    
    Args:
        text: Текст для анализа
        
    Returns:
        'past', 'future', 'current' или None
    """
    text_lower = text.lower()
    
    for temporal_type, patterns in TEMPORAL_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text_lower):
                return temporal_type
    
    return 'current'  # По умолчанию считаем текущим


def normalize_value(value: str, fact_type: FactType) -> str:
    """
    Нормализует извлеченное значение
    
    Args:
        value: Исходное значение
        fact_type: Тип факта
        
    Returns:
        Нормализованное значение
    """
    value = value.strip()
    
    # Убираем лишние пробелы
    value = re.sub(r'\s+', ' ', value)
    
    # Специфичная нормализация по типам
    if fact_type in [FactType.PERSONAL_NAME, FactType.FAMILY_SPOUSE, 
                     FactType.PET_NAME, FactType.PERSONAL_LOCATION]:
        # Капитализация имен и названий
        value = value.title()
    
    elif fact_type == FactType.PERSONAL_AGE:
        # Оставляем только число
        match = re.search(r'\d+', value)
        if match:
            value = match.group()
    
    elif fact_type in [FactType.PET_TYPE, FactType.SPORT_TYPE]:
        # Приводим к нижнему регистру
        value = value.lower()
    
    elif fact_type in [FactType.FINANCE_INCOME, FactType.FINANCE_SAVINGS, 
                       FactType.FINANCE_CREDIT, FactType.FINANCE_MORTGAGE]:
        # Убираем пробелы в числах
        value = re.sub(r'\s+', '', value)
    
    # Убираем знаки препинания в конце
    value = value.rstrip('.,!?;:')
    
    return value


def confidence_from_pattern_match(pattern_index: int, total_patterns: int) -> float:
    """
    Рассчитывает уверенность на основе индекса паттерна
    
    Args:
        pattern_index: Индекс сработавшего паттерна (0 = самый точный)
        total_patterns: Общее количество паттернов
        
    Returns:
        Уверенность от 0.5 до 1.0
    """
    if total_patterns <= 1:
        return 0.9
    
    # Линейное убывание уверенности
    base_confidence = 1.0
    decrease_step = 0.3 / total_patterns
    confidence = base_confidence - (pattern_index * decrease_step)
    
    return max(0.5, min(1.0, confidence))


def get_relation_for_type(fact_type: FactType) -> FactRelation:
    """Определяет отношение для типа факта"""
    relation_map = {
        # Личная информация
        FactType.PERSONAL_NAME: FactRelation.IS,
        FactType.PERSONAL_NICKNAME: FactRelation.IS,
        FactType.PERSONAL_AGE: FactRelation.IS,
        FactType.PERSONAL_LOCATION: FactRelation.LIVES_IN,
        FactType.PERSONAL_HOMETOWN: FactRelation.BORN_IN,
        
        # Работа
        FactType.WORK_OCCUPATION: FactRelation.WORKS_AS,
        FactType.WORK_COMPANY: FactRelation.WORKS_AT,
        FactType.WORK_SALARY: FactRelation.EARNS,
        
        # Транспорт
        FactType.TRANSPORT_CAR_BRAND: FactRelation.DRIVES,
        FactType.TRANSPORT_CAR_MODEL: FactRelation.DRIVES,
        
        # Финансы
        FactType.FINANCE_INCOME: FactRelation.EARNS,
        FactType.FINANCE_SAVINGS: FactRelation.SAVES,
        FactType.FINANCE_INVESTMENT: FactRelation.INVESTS_IN,
        FactType.FINANCE_BANK: FactRelation.BANKS_WITH,
        
        # Еда и напитки
        FactType.FOOD_FAVORITE: FactRelation.LIKES,
        FactType.FOOD_CUISINE: FactRelation.PREFERS,
        FactType.FOOD_RESTAURANT: FactRelation.VISITS,
        FactType.FOOD_CAFE: FactRelation.VISITS,
        FactType.DRINK_COFFEE: FactRelation.DRINKS,
        FactType.DRINK_TEA: FactRelation.DRINKS,
        FactType.DRINK_ALCOHOL: FactRelation.DRINKS,
        
        # Путешествия
        FactType.TRAVEL_COUNTRY: FactRelation.TRAVELS_TO,
        FactType.TRAVEL_CITY: FactRelation.TRAVELS_TO,
        FactType.TRAVEL_DREAM: FactRelation.PLANS_TO,
        
        # Спорт
        FactType.SPORT_TYPE: FactRelation.DOES,
        FactType.SPORT_GYM: FactRelation.TRAINS_AT,
        FactType.SPORT_TEAM: FactRelation.LIKES,
        
        # Семья
        FactType.FAMILY_SPOUSE: FactRelation.MARRIED_TO,
        FactType.FAMILY_CHILDREN: FactRelation.PARENT_OF,
        
        # Питомцы
        FactType.PET_NAME: FactRelation.OWNS,
        FactType.PET_TYPE: FactRelation.OWNS,
        FactType.PET_BREED: FactRelation.OWNS,
        
        # Хобби
        FactType.HOBBY_SPORT: FactRelation.DOES,
        FactType.HOBBY_ACTIVITY: FactRelation.DOES,
        FactType.HOBBY_GAMING: FactRelation.PLAYS,
        FactType.HOBBY_READING: FactRelation.READS,
        FactType.HOBBY_MUSIC: FactRelation.LISTENS_TO,
        FactType.HOBBY_INSTRUMENT: FactRelation.PLAYS,
        
        # Предпочтения
        FactType.PREFERENCE_FOOD: FactRelation.LIKES,
        FactType.PREFERENCE_MOVIE: FactRelation.WATCHES,
        FactType.PREFERENCE_BOOK: FactRelation.READS,
        FactType.PREFERENCE_BRAND: FactRelation.PREFERS,
        
        # Образование
        FactType.EDUCATION_INSTITUTION: FactRelation.STUDIED_AT,
        
        # Здоровье
        FactType.HEALTH_CONDITION: FactRelation.HAS,
        FactType.HEALTH_ALLERGY: FactRelation.HAS,
        
        # Недвижимость
        FactType.PROPERTY_TYPE: FactRelation.LIVES_IN,
        FactType.PROPERTY_OWNERSHIP: FactRelation.OWNS,
    }
    
    return relation_map.get(fact_type, FactRelation.IS)