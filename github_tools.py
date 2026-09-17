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
весь агент аварійно зупиниться.

Економія токен-бюджету: для .py файлів докстрінги видаляються ПЕРЕД
обрізанням за MAX_FILE_CHARS (через _strip_docstrings). Це свідомий
trade-off: агент бачить структуру коду й логіку обробки помилок, але
не оцінює якість самої документації - натомість менше файлів
обрізається посередині, і токен-бюджет витрачається на код, а не на
текст докстрінгів.
"""

import ast
import base64
import re
from typing import Any

import requests
from langchain_core.tools import tool

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 15
MAX_FILE_CHARS = 5000


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

    Args:
        url: Повний URL для запиту до GitHub API.

    Returns:
        Об'єкт відповіді requests.Response.
    """
    return requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)


def _strip_docstrings(source: str) -> str:
    """Видаляє докстрінги з Python-коду для економії токен-бюджету.

    Args:
        source: Вихідний Python-код як текст.

    Returns:
        Код без докстрінгів. Якщо код невалідний (SyntaxError) -
        повертає оригінал без змін.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())

    try:
        return ast.unparse(tree)
    except Exception:
        return source


@tool
def get_repo_info(repo_url: str) -> str:
    """Отримує загальну інформацію про репозиторій: опис, мову,
    зірки, дату оновлення. Виклич це ПЕРШИМ."""
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
    """Повертає список файлів/папок за шляхом (корінь за замовчуванням).
    Виклич це ДРУГИМ, щоб побачити структуру проєкту."""
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
    """Отримує вміст одного файлу (напр. 'main.py'). Виклич це ОДИН РАЗ
    для точки входу програми, щоб оцінити обробку помилок."""
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

        if file_path.endswith(".py"):
            content = _strip_docstrings(content)

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
