"""
Тести для analyze_repo() з main.py - перевіряють обробку різних
результатів від агента, включно з edge-case'ами, знайденими під час
код-рев'ю: порожній result["messages"] та відсутній content.

Мокуємо весь build_agent(), щоб не робити реальних викликів до
Groq API під час тестів.
"""

from unittest.mock import MagicMock, patch

import pytest

from main import analyze_repo


def _make_mock_agent(messages_to_return):
    """Створює мок-агента, invoke() якого повертає заданий список
    повідомлень у форматі, що очікує analyze_repo()."""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": messages_to_return}
    return mock_agent


@patch("main.build_agent")
def test_analyze_repo_returns_content_on_normal_response(mock_build_agent):
    """Перевіряємо звичайний, "щасливий" сценарій: агент повертає
    список повідомлень з валідним фінальним текстом."""
    final_message = MagicMock()
    final_message.content = "Сильні сторони: ... Загальна оцінка: ..."
    mock_build_agent.return_value = _make_mock_agent([final_message])

    result = analyze_repo("owner/repo")

    assert result == "Сильні сторони: ... Загальна оцінка: ..."


@patch("main.build_agent")
def test_analyze_repo_raises_on_empty_messages_list(mock_build_agent):
    """Перевіряємо edge-case, знайдений під час код-рев'ю: якщо
    result["messages"] порожній список - раніше це впало б з
    IndexError. Тепер має піднятись зрозумілий RuntimeError."""
    mock_build_agent.return_value = _make_mock_agent([])

    with pytest.raises(RuntimeError, match="не повернув жодної текстової відповіді"):
        analyze_repo("owner/repo")


@patch("main.build_agent")
def test_analyze_repo_raises_when_content_is_empty_string(mock_build_agent):
    """Перевіряємо інший edge-case: повідомлення є, але content -
    порожній рядок (наприклад, через замалий max_tokens для
    reasoning-моделі - той самий баг, який ми реально бачили раніше)."""
    final_message = MagicMock()
    final_message.content = ""
    mock_build_agent.return_value = _make_mock_agent([final_message])

    with pytest.raises(RuntimeError, match="не повернув жодної текстової відповіді"):
        analyze_repo("owner/repo")


@patch("main.build_agent")
def test_analyze_repo_raises_when_content_is_none(mock_build_agent):
    """Перевіряємо, що None замість content теж коректно обробляється,
    а не викликає AttributeError десь глибше в коді."""
    final_message = MagicMock()
    final_message.content = None
    mock_build_agent.return_value = _make_mock_agent([final_message])

    with pytest.raises(RuntimeError, match="не повернув жодної текстової відповіді"):
        analyze_repo("owner/repo")
