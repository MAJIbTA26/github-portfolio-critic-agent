"""
GitHub Portfolio Critic Agent

AI-агент, що аналізує публічний GitHub-репозиторій та дає структурований
фідбек про якість коду й документації - схожий на той фідбек, який
реально отримав автор цього проєкту від компанії-роботодавця.

Це справжній ReAct-агент (Reasoning + Acting): LLM сам вирішує,
у якому порядку викликати інструменти (get_repo_info -> list_repo_files ->
get_file_content для конкретних файлів), а не виконує жорстко заданий
сценарій.

Запуск:
    python main.py
"""

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from github_tools import get_repo_info, list_repo_files, get_file_content

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ти - досвідчений технічний рев'ювер, що аналізує
pet-проєкти на GitHub так, як це робить технічний лід перед прийняттям
рішення про наймання розробника.

Твій процес аналізу (СТРОГО обмежений, не виходь за ці рамки):
1. Виклич get_repo_info ОДИН раз, щоб зрозуміти загальний контекст.
2. Виклич list_repo_files ОДИН раз (тільки для кореневої папки, path=""),
   щоб побачити структуру проєкту.
3. ОБОВ'ЯЗКОВО виклич get_file_content для головного файлу застосунку
   (зазвичай main.py, app.py або bot.py - подивись на список файлів і
   визнач сам, який із них є точкою входу) - саме там зазвичай міститься
   обробка помилок, яку важливо оцінити.
4. Додатково (опційно) виклич get_file_content ЩЕ для 1 файлу - README
   АБО одного допоміжного модуля з кодом, якщо вважаєш це важливим.
   Не більше 2 додаткових викликів get_file_content разом.

Після цього ОБОВ'ЯЗКОВО дай фінальний фідбек текстом, БЕЗ подальших
викликів інструментів, за такою структурою:
- **Сильні сторони** (2-3 пункти)
- **Що варто покращити** (2-3 конкретні, дієві поради)
- **Загальна оцінка** (1 речення: чи виглядає це як робота людини,
  готової до Junior-позиції)

Будь конкретним і чесним. Не досліджуй репозиторій більше, ніж описано
вище - обмежена кількість викликів інструментів є навмисною."""


def build_agent():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY не знайдено. Створи .env файл на основі "
            ".env.example і встав свій ключ з console.groq.com"
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0,
        max_tokens=2048,  # reasoning-модель витрачає токени на приховані
                            # роздуми - замалий ліміт міг спричиняти обірвані
                            # відповіді без tool_calls і без фінального тексту
    )

    tools = [get_repo_info, list_repo_files, get_file_content]
    agent = create_react_agent(llm, tools)
    return agent


def analyze_repo(repo_url: str) -> str:
    agent = build_agent()

    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Проаналізуй цей репозиторій: {repo_url}"}
            ]
        },
        config={"recursion_limit": 50},  # запас про всяк випадок, поверх
                                            # обмежень у самому промті
    )

    # Останнє повідомлення в результаті - фінальна відповідь агента
    final_message = result["messages"][-1]
    return final_message.content


def main():
    print("=== GitHub Portfolio Critic Agent ===\n")
    repo_url = input("Встав посилання на GitHub-репозиторій: ").strip()

    if not repo_url:
        print("Посилання не вказано, завершую роботу.")
        return

    logger.info(f"Починаю аналіз репозиторію: {repo_url}")
    print(f"\nАналізую {repo_url}...\n")
    print("(агент сам вирішує, які інструменти викликати - це може зайняти кілька запитів)\n")

    try:
        feedback = analyze_repo(repo_url)
        logger.info("Аналіз завершено успішно")
        print("=" * 60)
        print(feedback)
        print("=" * 60)

    except RuntimeError as e:
        # Відсутній GROQ_API_KEY - див. build_agent()
        logger.error(f"Помилка конфігурації: {e}")
        print(f"\n❌ Помилка конфігурації: {e}")

    except ValueError as e:
        # Некоректний формат посилання - див. _parse_repo_url()
        logger.error(f"Некоректне посилання: {e}")
        print(f"\n❌ Некоректне посилання на репозиторій: {e}")

    except requests.exceptions.ConnectionError:
        logger.error("Немає з'єднання з інтернетом")
        print("\n❌ Немає з'єднання з інтернетом. Перевір мережу і спробуй ще раз.")

    except requests.exceptions.Timeout:
        logger.error("GitHub API не відповів вчасно")
        print("\n❌ GitHub API не відповів вчасно (timeout). Спробуй ще раз.")

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 403:
            logger.error("GitHub API: перевищено ліміт запитів без токена")
            print(
                "\n❌ GitHub API повернув 403 - найімовірніше, перевищено ліміт "
                "60 запитів/год без токена. Зачекай трохи, або додай GitHub "
                "Personal Access Token для вищого ліміту."
            )
        elif status == 429:
            logger.error("Перевищено rate limit API")
            print("\n❌ Перевищено ліміт запитів (429). Зачекай хвилину і спробуй ще раз.")
        else:
            logger.error(f"HTTP помилка {status}: {e}")
            print(f"\n❌ HTTP помилка від API ({status}): {e}")

    except Exception as e:
        # Все інше (помилки Groq API, несподівані збої тощо) - не приховуємо,
        # але хоча б логуємо з повним типом помилки для діагностики.
        logger.error(f"Неочікувана помилка ({type(e).__name__}): {e}")
        print(f"\n❌ Неочікувана помилка ({type(e).__name__}): {e}")
        print("Якщо це повторюється - перевір GROQ_API_KEY та ліміти на console.groq.com")


if __name__ == "__main__":
    main()
