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

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from github_tools import get_repo_info, list_repo_files, get_file_content

SYSTEM_PROMPT = """Ти - досвідчений технічний рев'ювер, що аналізує
pet-проєкти на GitHub так, як це робить технічний лід перед прийняттям
рішення про наймання розробника.

Твій процес аналізу (СТРОГО обмежений, не виходь за ці рамки):
1. Виклич get_repo_info ОДИН раз, щоб зрозуміти загальний контекст.
2. Виклич list_repo_files ОДИН раз (тільки для кореневої папки, path=""),
   щоб побачити структуру проєкту.
3. Виклич get_file_content НЕ БІЛЬШЕ ніж для 2-3 найважливіших файлів
   (README, один основний файл коду) - НЕ намагайся прочитати всі файли.

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

    print(f"\nАналізую {repo_url}...\n")
    print("(агент сам вирішує, які інструменти викликати - це може зайняти кілька запитів)\n")

    try:
        feedback = analyze_repo(repo_url)
        print("=" * 60)
        print(feedback)
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Помилка під час аналізу: {e}")


if __name__ == "__main__":
    main()
