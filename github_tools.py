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
import json
import logging
import re
from typing import Any

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

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
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
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


def _analyze_python_file(filename: str, source: str) -> dict[str, Any]:
    """Аналізує один Python-файл через AST і повертає компактні
    метрики якості коду - БЕЗ виклику LLM, тільки статичний аналіз.

    Це ключова частина Map-Reduce підходу: замість того щоб надсилати
    LLM повний вміст кожного файлу (дорого за токенами, і розмір росте
    лінійно з розміром файлу), рахуємо метрики локально, і розмір
    результату залишається компактним НЕЗАЛЕЖНО від розміру файлу.

    Args:
        filename: Назва файлу (для ідентифікації в результаті).
        source: Вихідний Python-код як текст.

    Returns:
        Словник з метриками: кількість рядків/функцій/класів, відсоток
        функцій з докстрінгами/type hints, чи є logging/try-except.
        Якщо файл невалідний Python - повертає {"filename": ..., "error": ...}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"filename": filename, "error": f"SyntaxError: {e}"}

    functions = []
    classes = []
    has_logging_import = False
    has_try_except = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging":
                    has_logging_import = True
        if isinstance(node, ast.Try):
            has_try_except = True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            has_docstring = bool(
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            has_type_hints = node.returns is not None or any(
                arg.annotation is not None for arg in node.args.args
            )
            functions.append({
                "name": node.name,
                "has_docstring": has_docstring,
                "has_type_hints": has_type_hints,
            })
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

    total_funcs = len(functions)
    funcs_with_docstrings = sum(1 for f in functions if f["has_docstring"])
    funcs_with_types = sum(1 for f in functions if f["has_type_hints"])

    return {
        "filename": filename,
        "lines_of_code": len(source.splitlines()),
        "functions_count": total_funcs,
        "classes_count": len(classes),
        "docstring_coverage_pct": (
            round(funcs_with_docstrings / total_funcs * 100) if total_funcs else None
        ),
        "type_hints_coverage_pct": (
            round(funcs_with_types / total_funcs * 100) if total_funcs else None
        ),
        "has_logging": has_logging_import,
        "has_try_except": has_try_except,
        "function_names": [f["name"] for f in functions],
    }


@tool
def get_project_metrics(repo_url: str) -> str:
    """Аналізує ВСІ .py файли в корені репозиторію ОДНИМ викликом:
    докстрінги, типізація, логування, обробка помилок. Виклич це
    замість get_file_content, щоб побачити весь проєкт компактно."""
    try:
        owner, repo = _parse_repo_url(repo_url)
        logger.info(f"get_project_metrics: аналізую {owner}/{repo}")
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/")

        if response.status_code == 404:
            logger.warning(f"get_project_metrics: репозиторій {owner}/{repo} не знайдено")
            return f"Помилка: репозиторій '{owner}/{repo}' не знайдено."
        if response.status_code == 403:
            logger.warning("get_project_metrics: перевищено ліміт запитів GitHub API")
            return "Помилка: перевищено ліміт запитів до GitHub API (403)."
        response.raise_for_status()

        items: list[dict[str, Any]] = response.json()
        py_files = [
            item["name"] for item in items
            if item.get("type") == "file" and item["name"].endswith(".py")
        ]

        if not py_files:
            logger.info(f"get_project_metrics: у {owner}/{repo} немає .py файлів у корені")
            return "У кореневій папці немає .py файлів."

        all_metrics = []
        for filename in py_files:
            file_response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{filename}")
            if file_response.status_code != 200:
                logger.warning(f"get_project_metrics: не вдалось завантажити {filename}")
                all_metrics.append({"filename": filename, "error": "не вдалось завантажити"})
                continue

            file_data: dict[str, Any] = file_response.json()
            if file_data.get("encoding") != "base64":
                continue

            content = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
            metrics = _analyze_python_file(filename, content)
            all_metrics.append(metrics)

        logger.info(f"get_project_metrics: успішно проаналізовано {len(all_metrics)} файлів")
        return json.dumps(all_metrics, ensure_ascii=False, indent=2)

    except ValueError as e:
        logger.error(f"get_project_metrics: помилка формату посилання: {e}")
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        logger.error("get_project_metrics: немає з'єднання з інтернетом")
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        logger.error("get_project_metrics: GitHub API timeout")
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        logger.error(f"get_project_metrics: HTTP помилка: {e}")
        return f"Помилка HTTP від GitHub API: {e}"


@tool
def get_repo_info(repo_url: str) -> str:
    """Отримує загальну інформацію про репозиторій: опис, мову,
    зірки, дату оновлення. Виклич це ПЕРШИМ."""
    try:
        owner, repo = _parse_repo_url(repo_url)
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}")

        if response.status_code == 404:
            logger.warning(f"get_repo_info: репозиторій {owner}/{repo} не знайдено")
            return f"Помилка: репозиторій '{owner}/{repo}' не знайдено (можливо, приватний або не існує)."
        if response.status_code == 403:
            logger.warning("get_repo_info: перевищено ліміт запитів GitHub API")
            return "Помилка: перевищено ліміт запитів до GitHub API (403). Спробуй пізніше."
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        logger.info(f"get_repo_info: успішно отримано інфо про {owner}/{repo}")
        return (
            f"Назва: {data.get('full_name')}\n"
            f"Опис: {data.get('description') or '(немає опису)'}\n"
            f"Мова: {data.get('language') or 'не визначено'}\n"
            f"Зірки: {data.get('stargazers_count')}\n"
            f"Останнє оновлення: {data.get('updated_at')}\n"
            f"Розмір: {data.get('size')} KB"
        )

    except ValueError as e:
        logger.error(f"get_repo_info: помилка формату посилання: {e}")
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        logger.error("get_repo_info: немає з'єднання з інтернетом")
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        logger.error("get_repo_info: GitHub API timeout")
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        logger.error(f"get_repo_info: HTTP помилка: {e}")
        return f"Помилка HTTP від GitHub API: {e}"


@tool
def list_repo_files(repo_url: str, path: str = "") -> str:
    """Повертає список файлів/папок за шляхом (корінь за замовчуванням).
    Виклич це ДРУГИМ, щоб побачити структуру проєкту."""
    try:
        owner, repo = _parse_repo_url(repo_url)
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}")

        if response.status_code == 404:
            logger.warning(f"list_repo_files: шлях '{path}' не знайдено в {owner}/{repo}")
            return f"Помилка: шлях '{path}' не знайдено в репозиторії."
        if response.status_code == 403:
            logger.warning("list_repo_files: перевищено ліміт запитів GitHub API")
            return "Помилка: перевищено ліміт запитів до GitHub API (403). Спробуй пізніше."
        response.raise_for_status()

        items: list[dict[str, Any]] = response.json()
        if not isinstance(items, list):
            return f"'{path}' - це файл, а не папка. Використай get_file_content."

        lines: list[str] = []
        for item in items:
            marker = "📁" if item["type"] == "dir" else "📄"
            lines.append(f"{marker} {item['name']}")
        logger.info(f"list_repo_files: знайдено {len(lines)} елементів у {owner}/{repo}/{path}")
        return "\n".join(lines) if lines else "(порожня папка)"

    except ValueError as e:
        logger.error(f"list_repo_files: помилка формату посилання: {e}")
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        logger.error("list_repo_files: немає з'єднання з інтернетом")
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        logger.error("list_repo_files: GitHub API timeout")
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        logger.error(f"list_repo_files: HTTP помилка: {e}")
        return f"Помилка HTTP від GitHub API: {e}"


@tool
def get_file_content(repo_url: str, file_path: str) -> str:
    """Отримує вміст одного файлу (напр. 'main.py'). Виклич це ОДИН РАЗ
    для точки входу програми, щоб оцінити обробку помилок."""
    try:
        owner, repo = _parse_repo_url(repo_url)
        response = _safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path}")

        if response.status_code == 404:
            logger.warning(f"get_file_content: файл '{file_path}' не знайдено в {owner}/{repo}")
            return f"Помилка: файл '{file_path}' не знайдено."
        if response.status_code == 403:
            logger.warning("get_file_content: перевищено ліміт запитів GitHub API")
            return "Помилка: перевищено ліміт запитів до GitHub API (403). Спробуй пізніше."
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        if data.get("encoding") != "base64":
            logger.warning(f"get_file_content: незвичне кодування у '{file_path}'")
            return f"Не вдалось прочитати файл '{file_path}' (незвичне кодування)."

        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

        if file_path.endswith(".py"):
            content = _strip_docstrings(content)

        if len(content) > MAX_FILE_CHARS:
            logger.info(f"get_file_content: '{file_path}' обрізано ({len(content)} -> {MAX_FILE_CHARS} символів)")
            content = content[:MAX_FILE_CHARS] + "\n... (файл обрізано, занадто довгий)"
        else:
            logger.info(f"get_file_content: успішно отримано '{file_path}' ({len(content)} символів)")

        return content

    except ValueError as e:
        logger.error(f"get_file_content: помилка формату посилання: {e}")
        return f"Помилка формату посилання: {e}"
    except requests.exceptions.ConnectionError:
        logger.error("get_file_content: немає з'єднання з інтернетом")
        return "Помилка: немає з'єднання з інтернетом."
    except requests.exceptions.Timeout:
        logger.error("get_file_content: GitHub API timeout")
        return "Помилка: GitHub API не відповів вчасно (timeout)."
    except requests.exceptions.HTTPError as e:
        logger.error(f"get_file_content: HTTP помилка: {e}")
        return f"Помилка HTTP від GitHub API: {e}"
