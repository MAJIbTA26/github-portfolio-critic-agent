"""
Інструменти (tools) для агента-критика GitHub-репозиторіїв.

Кожна функція обгорнута декоратором @tool з langchain - це перетворює
звичайну Python-функцію на "інструмент", який LLM-агент може сам
викликати, коли вирішить, що це потрібно для відповіді на запит.

Використовує публічне GitHub REST API - для читання публічних
репозиторіїв токен не потрібен (є ліміт 60 запитів/год без токена,
цього достатньо для аналізу одного репозиторію).

Обробка помилок: кожен tool ловить мережеві помилки САМОСТІЙНО і
повертає їх як текстове повідомлення (а не піднімає виняток). Це
важливо для агента: якщо tool впаде з винятком посеред ReAct-циклу,
весь агент аварійно зупиниться. Натомість, коли tool повертає текст
на кшталт "Помилка: немає з'єднання" - агент бачить це як звичайний
результат і може або спробувати ще раз, або повідомити про проблему
користувачу в фінальній відповіді.
"""

import base64
import re
from typing import Any

import requests
from langchain_core.tools import tool

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 15
MAX_FILE_CHARS = 3000


def _parse_repo_url(url_or_path: str) -> tuple[str, str]:
    """Витягує (owner, repo) з різних форматів посилання на GitHub.

    Args:
        url_or_path: Посилання у будь-якому з форматів:
            "https://github.com/owner/repo",
            "github.com/owner/repo",
            "owner/repo".

    Returns:
        Кортеж (owner, repo) - власник репозиторію та його назва.

    Raises:
        ValueError: якщо рядок не вдалось розпізнати як owner/repo.
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


def _safe_get(url: str) -> requests.Response:
    """Виконує GET-запит до GitHub API.

    Внутрішня допоміжна функція (не tool) - централізує сам виклик
    requests.get з таймаутом, щоб не дублювати параметри в кожному
    з трьох tools нижче. Мережеві винятки НЕ ловляться тут навмисно -
    їх ловить викликаючий tool, щоб повернути агенту зрозумілий текст
    замість аварійного завершення.

    Args:
        url: Повний URL для запиту до GitHub API.

    Returns:
        Об'єкт відповіді requests.Response.
    """
    return requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)


@tool
def get_repo_info(repo_url: str) -> str:
    """Отримує загальну інформацію про GitHub-репозиторій: опис, основну
    мову програмування, кількість зірок, дату останнього оновлення.
    Використовуй це ПЕРШИМ, щоб отримати загальний контекст перед
    детальнішим аналізом.

    Args:
        repo_url: Посилання на репозиторій (owner/repo або повний URL).

    Returns:
        Текстовий опис репозиторію, або повідомлення про помилку,
        якщо репозиторій не знайдено чи виникла мережева проблема.
    """
    try:
        owner, repo = _parse_repo_url(repo_url)
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}")

        if response.status_code == 404:
            return f"Помилка: репозиторій '{owner}/{repo}' не знайдено (можливо, приватний або не існує)."
        if response.status_code == 403:
            return "Помилка: перевищено ліміт запитів до GitHub API (403). Спробуй пізніше."
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        return (
            f"Назва: {data.get('full_name')}\n"
            f"Опис: {data.get('description') or '(немає опису)'}\n"
            f"Мова: {data.get('language') or 'не визначено'}\n"
            f"Зірки: {data.get('stargazers_count')}\n"
            f"Останнє оновлення: {data.get('updated_at')}\n"
            f"Розмір: {data.get('size')} KB"
        )

    except ValueError as e:
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        return f"Помилка HTTP від GitHub API: {e}"


@tool
def list_repo_files(repo_url: str, path: str = "") -> str:
    """Повертає список файлів та папок у репозиторії за вказаним шляхом
    (кореневий шлях за замовчуванням). Використовуй це, щоб побачити
    структуру проєкту перед тим, як вирішити, у які саме файли заглянути
    детальніше.

    Args:
        repo_url: Посилання на репозиторій (owner/repo або повний URL).
        path: Шлях усередині репозиторію (порожній рядок = корінь).

    Returns:
        Список файлів/папок у вигляді тексту (по одному на рядок,
        з іконками 📁/📄), або повідомлення про помилку.
    """
    try:
        owner, repo = _parse_repo_url(repo_url)
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}")

        if response.status_code == 404:
            return f"Помилка: шлях '{path}' не знайдено в репозиторії."
        if response.status_code == 403:
            return "Помилка: перевищено ліміт запитів до GitHub API (403). Спробуй пізніше."
        response.raise_for_status()

        items: list[dict[str, Any]] = response.json()
        if not isinstance(items, list):
            return f"'{path}' - це файл, а не папка. Використай get_file_content."

        lines: list[str] = []
        for item in items:
            marker = "📁" if item["type"] == "dir" else "📄"
            lines.append(f"{marker} {item['name']}")
        return "\n".join(lines) if lines else "(порожня папка)"

    except ValueError as e:
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        return f"Помилка HTTP від GitHub API: {e}"


@tool
def get_file_content(repo_url: str, file_path: str) -> str:
    """Отримує вміст конкретного файлу з репозиторію (наприклад,
    'main.py', 'requirements.txt', 'README.md'). Використовуй це, коли
    хочеш перевірити конкретну деталь: чи є обробка помилок, чи є тести,
    чи є .gitignore тощо. Не запитуй занадто великі файли повністю без
    потреби.

    Args:
        repo_url: Посилання на репозиторій (owner/repo або повний URL).
        file_path: Шлях до файлу всередині репозиторію.

    Returns:
        Вміст файлу як текст (обрізаний до MAX_FILE_CHARS символів,
        якщо файл занадто довгий), або повідомлення про помилку.
    """
    try:
        owner, repo = _parse_repo_url(repo_url)
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path}")

        if response.status_code == 404:
            return f"Помилка: файл '{file_path}' не знайдено."
        if response.status_code == 403:
            return "Помилка: перевищено ліміт запитів до GitHub API (403). Спробуй пізніше."
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        if data.get("encoding") != "base64":
            return f"Не вдалось прочитати файл '{file_path}' (незвичне кодування)."

        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n... (файл обрізано, занадто довгий)"

        return content

    except ValueError as e:
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        return f"Помилка HTTP від GitHub API: {e}"
