"""
Автотести для github_tools.py.

Використовує unittest.mock, щоб замінити реальні мережеві виклики
(requests.get) на контрольовані, передбачувані відповіді - тести
працюють миттєво, без інтернету, і не залежать від стану реального
GitHub API чи від rate limits.

Запуск:
    pytest test_github_tools.py -v
"""

from unittest.mock import Mock, patch

import pytest
import requests

from github_tools import (
    _parse_repo_url,
    get_file_content,
    get_repo_info,
    list_repo_files,
)

# ---------------------------------------------------------------------
# Тести для _parse_repo_url (без моків - чиста логіка, без мережі)
# ---------------------------------------------------------------------

def test_parse_repo_url_full_https_link():
    owner, repo = _parse_repo_url("https://github.com/MAJIbTA26/test-repo")
    assert owner == "MAJIbTA26"
    assert repo == "test-repo"


def test_parse_repo_url_without_protocol():
    owner, repo = _parse_repo_url("github.com/MAJIbTA26/test-repo")
    assert owner == "MAJIbTA26"
    assert repo == "test-repo"


def test_parse_repo_url_short_form():
    owner, repo = _parse_repo_url("MAJIbTA26/test-repo")
    assert owner == "MAJIbTA26"
    assert repo == "test-repo"


def test_parse_repo_url_trailing_slash():
    owner, repo = _parse_repo_url("https://github.com/MAJIbTA26/test-repo/")
    assert owner == "MAJIbTA26"
    assert repo == "test-repo"


def test_parse_repo_url_raises_on_invalid_input():
    with pytest.raises(ValueError):
        _parse_repo_url("тільки-одне-слово-без-власника")


# ---------------------------------------------------------------------
# Тести для get_repo_info (з моком requests.get)
# ---------------------------------------------------------------------

@patch("github_tools.requests.get")
def test_get_repo_info_returns_formatted_data(mock_get):
    """Перевіряємо, що успішна відповідь API коректно форматується."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "full_name": "MAJIbTA26/test-repo",
        "description": "Тестовий опис",
        "language": "Python",
        "stargazers_count": 5,
        "updated_at": "2026-01-01T00:00:00Z",
        "size": 120,
    }
    mock_get.return_value = mock_response

    result = get_repo_info.invoke({"repo_url": "MAJIbTA26/test-repo"})

    assert "MAJIbTA26/test-repo" in result
    assert "Python" in result
    assert "5" in result


@patch("github_tools.requests.get")
def test_get_repo_info_handles_404(mock_get):
    """Перевіряємо, що неіснуючий репозиторій дає зрозуміле повідомлення,
    а не падає з винятком."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = get_repo_info.invoke({"repo_url": "MAJIbTA26/repo-that-does-not-exist"})

    assert "не знайдено" in result.lower()


@patch("github_tools.requests.get")
def test_get_repo_info_handles_connection_error(mock_get):
    """Перевіряємо, що відсутність інтернету дає зрозуміле повідомлення,
    а не аварійно завершує програму."""
    mock_get.side_effect = requests.exceptions.ConnectionError()

    result = get_repo_info.invoke({"repo_url": "MAJIbTA26/test-repo"})

    assert "з'єднання" in result.lower()


@patch("github_tools.requests.get")
def test_get_repo_info_handles_timeout(mock_get):
    """Перевіряємо обробку таймауту."""
    mock_get.side_effect = requests.exceptions.Timeout()

    result = get_repo_info.invoke({"repo_url": "MAJIbTA26/test-repo"})

    assert "timeout" in result.lower() or "не відповів" in result.lower()


# ---------------------------------------------------------------------
# Тести для list_repo_files
# ---------------------------------------------------------------------

@patch("github_tools.requests.get")
def test_list_repo_files_formats_files_and_folders(mock_get):
    """Перевіряємо, що файли і папки правильно позначаються іконками."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"name": "main.py", "type": "file"},
        {"name": "tests", "type": "dir"},
    ]
    mock_get.return_value = mock_response

    result = list_repo_files.invoke({"repo_url": "MAJIbTA26/test-repo", "path": ""})

    assert "📄 main.py" in result
    assert "📁 tests" in result


@patch("github_tools.requests.get")
def test_list_repo_files_handles_empty_folder(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    result = list_repo_files.invoke({"repo_url": "MAJIbTA26/test-repo", "path": "empty"})

    assert "порожня" in result.lower()


# ---------------------------------------------------------------------
# Тести для get_file_content
# ---------------------------------------------------------------------

@patch("github_tools.requests.get")
def test_get_file_content_decodes_base64(mock_get):
    """Перевіряємо, що base64-контент правильно декодується."""
    import base64

    original_text = "print('Hello, world!')"
    encoded = base64.b64encode(original_text.encode("utf-8")).decode("ascii")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"encoding": "base64", "content": encoded}
    mock_get.return_value = mock_response

    result = get_file_content.invoke({"repo_url": "MAJIbTA26/test-repo", "file_path": "main.py"})

    assert result == original_text


@patch("github_tools.requests.get")
def test_get_file_content_truncates_long_files(mock_get):
    """Перевіряємо, що занадто довгі файли обрізаються."""
    import base64

    long_text = "x" * 5000  # довше за MAX_FILE_CHARS (3000)
    encoded = base64.b64encode(long_text.encode("utf-8")).decode("ascii")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"encoding": "base64", "content": encoded}
    mock_get.return_value = mock_response

    result = get_file_content.invoke({"repo_url": "MAJIbTA26/test-repo", "file_path": "big_file.py"})

    assert len(result) < 5000
    assert "обрізано" in result
