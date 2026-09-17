"""
Тести для get_project_metrics() та _analyze_python_file() - ключових
функцій Map-Reduce архітектури, яка замінила читання сирих файлів на
компактний статичний аналіз через AST.
"""

import base64
import json
from unittest.mock import Mock, patch

from github_tools import _analyze_python_file, get_project_metrics


def test_analyze_python_file_detects_good_practices():
    """Перевіряємо, що файл з докстрінгами, типізацією, логуванням
    та обробкою помилок отримує 100% покриття по відповідних метриках."""
    code = '''
import logging
logger = logging.getLogger(__name__)

def build_agent() -> str:
    """Docstring."""
    try:
        return "ok"
    except Exception:
        pass
'''
    result = _analyze_python_file("main.py", code)

    assert result["filename"] == "main.py"
    assert result["functions_count"] == 1
    assert result["docstring_coverage_pct"] == 100
    assert result["type_hints_coverage_pct"] == 100
    assert result["has_logging"] is True
    assert result["has_try_except"] is True


def test_analyze_python_file_detects_missing_practices():
    """Перевіряємо, що файл без докстрінгів/типізації/логування
    коректно отримує 0% і False по відповідних метриках."""
    code = '''
def helper(x):
    return x + 1
'''
    result = _analyze_python_file("helper.py", code)

    assert result["docstring_coverage_pct"] == 0
    assert result["type_hints_coverage_pct"] == 0
    assert result["has_logging"] is False
    assert result["has_try_except"] is False


def test_analyze_python_file_handles_syntax_error():
    """Перевіряємо, що невалідний Python-код не ламає аналіз, а
    повертає зрозуміле повідомлення про помилку."""
    broken_code = "def broken(:\n    return"

    result = _analyze_python_file("broken.py", broken_code)

    assert "error" in result
    assert "SyntaxError" in result["error"]


def test_analyze_python_file_handles_file_with_no_functions():
    """Перевіряємо граничний випадок: файл без жодної функції
    (наприклад, лише константи) не викликає ділення на нуль."""
    code = "X = 1\nY = 2\n"

    result = _analyze_python_file("constants.py", code)

    assert result["functions_count"] == 0
    assert result["docstring_coverage_pct"] is None
    assert result["type_hints_coverage_pct"] is None


@patch("github_tools.requests.get")
def test_get_project_metrics_analyzes_multiple_files_and_skips_non_python(mock_get):
    """Перевіряємо повний цикл get_project_metrics: кілька .py файлів
    аналізуються, README.md ігнорується, результат - валідний JSON."""

    good_code = 'def f() -> int:\n    """Doc."""\n    return 1\n'
    bad_code = "def g(x):\n    return x\n"

    def fake_get(url, timeout=None):
        resp = Mock()
        resp.status_code = 200
        if url.endswith("/contents/"):
            resp.json.return_value = [
                {"name": "main.py", "type": "file"},
                {"name": "helper.py", "type": "file"},
                {"name": "README.md", "type": "file"},
            ]
        elif "main.py" in url:
            resp.json.return_value = {
                "encoding": "base64",
                "content": base64.b64encode(good_code.encode()).decode(),
            }
        elif "helper.py" in url:
            resp.json.return_value = {
                "encoding": "base64",
                "content": base64.b64encode(bad_code.encode()).decode(),
            }
        return resp

    mock_get.side_effect = fake_get

    result = get_project_metrics.invoke({"repo_url": "owner/repo"})
    parsed = json.loads(result)

    filenames = [item["filename"] for item in parsed]
    assert "main.py" in filenames
    assert "helper.py" in filenames
    assert "README.md" not in filenames  # не .py файл - має бути проігнорований
    assert len(parsed) == 2


@patch("github_tools.requests.get")
def test_get_project_metrics_returns_message_when_no_python_files(mock_get):
    """Перевіряємо граничний випадок: репозиторій без жодного .py файлу."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"name": "README.md", "type": "file"}]
    mock_get.return_value = mock_response

    result = get_project_metrics.invoke({"repo_url": "owner/repo"})

    assert "немає" in result.lower()
