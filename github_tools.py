"""
Інструменти (tools) для агента-критика GitHub-репозиторіїв.

Кожна функція обгорнута декоратором @tool з langchain - це перетворює
звичайну Python-функцію на "інструмент", який LLM-агент може сам
викликати, коли вирішить, що це потрібно для відповіді на запит.

Використовує публічне GitHub REST API - для читання публічних
репозиторіїв токен не потрібен (є ліміт 60 запитів/год без токена,
цього достатньо для аналізу одного репозиторію).
"""

import base64
import re
import requests
from langchain_core.tools import tool

GITHUB_API = "https://api.github.com"


def _parse_repo_url(url_or_path: str) -> tuple[str, str]:
    """Витягує (owner, repo) з різних форматів посилання на GitHub.

    Приймає:
        https://github.com/owner/repo
        github.com/owner/repo
        owner/repo
    """
    cleaned = url_or_path.strip().rstrip("/")
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = re.sub(r"^github\.com/", "", cleaned)
    parts = cleaned.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Не вдалось розпізнати власника та назву репозиторію з '{url_or_path}'. "
            f"Очікується формат 'owner/repo' або повне посилання на GitHub."
        )
    return parts[0], parts[1]


@tool
def get_repo_info(repo_url: str) -> str:
    """Отримує загальну інформацію про GitHub-репозиторій: опис, основну
    мову програмування, кількість зірок, дату останнього оновлення.
    Використовуй це ПЕРШИМ, щоб отримати загальний контекст перед
    детальнішим аналізом."""
    owner, repo = _parse_repo_url(repo_url)
    response = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}", timeout=15)

    if response.status_code == 404:
        return f"Помилка: репозиторій '{owner}/{repo}' не знайдено (можливо, приватний або не існує)."
    response.raise_for_status()

    data = response.json()
    return (
        f"Назва: {data.get('full_name')}\n"
        f"Опис: {data.get('description') or '(немає опису)'}\n"
        f"Мова: {data.get('language') or 'не визначено'}\n"
        f"Зірки: {data.get('stargazers_count')}\n"
        f"Останнє оновлення: {data.get('updated_at')}\n"
        f"Розмір: {data.get('size')} KB"
    )


@tool
def list_repo_files(repo_url: str, path: str = "") -> str:
    """Повертає список файлів та папок у репозиторії за вказаним шляхом
    (кореневий шлях за замовчуванням). Використовуй це, щоб побачити
    структуру проєкту перед тим, як вирішити, у які саме файли заглянути
    детальніше."""
    owner, repo = _parse_repo_url(repo_url)
    response = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}", timeout=15
    )

    if response.status_code == 404:
        return f"Помилка: шлях '{path}' не знайдено в репозиторії."
    response.raise_for_status()

    items = response.json()
    if not isinstance(items, list):
        return f"'{path}' - це файл, а не папка. Використай get_file_content."

    lines = []
    for item in items:
        marker = "📁" if item["type"] == "dir" else "📄"
        lines.append(f"{marker} {item['name']}")
    return "\n".join(lines) if lines else "(порожня папка)"


@tool
def get_file_content(repo_url: str, file_path: str) -> str:
    """Отримує вміст конкретного файлу з репозиторію (наприклад,
    'main.py', 'requirements.txt', 'README.md'). Використовуй це, коли
    хочеш перевірити конкретну деталь: чи є обробка помилок, чи є тести,
    чи є .gitignore тощо. Не запитуй занадто великі файли повністю без
    потреби."""
    owner, repo = _parse_repo_url(repo_url)
    response = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path}", timeout=15
    )

    if response.status_code == 404:
        return f"Помилка: файл '{file_path}' не знайдено."
    response.raise_for_status()

    data = response.json()
    if data.get("encoding") != "base64":
        return f"Не вдалось прочитати файл '{file_path}' (незвичне кодування)."

    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    # Обмежуємо довжину, щоб не витрачати забагато токенів на один файл
    max_chars = 3000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (файл обрізано, занадто довгий)"

    return content
