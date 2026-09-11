"""
Швидкий тест інструментів БЕЗ агента/LLM - просто перевіряє, що
GitHub API tools реально працюють на живому інтернеті.

Запусти це ПЕРШИМ, перед запуском main.py - якщо тут все ок,
значить проблема (якщо буде) точно в LLM-частині, а не в GitHub API.

Запуск:
    python test_tools_manually.py
"""

from github_tools import get_file_content, get_repo_info, list_repo_files

# Тест на реальному, публічному репозиторії - НЕ твій, а маленький
# офіційний приклад від GitHub, щоб перевірити базову працездатність
TEST_REPO = "octocat/Hello-World"

print(f"Тестую на репозиторії: {TEST_REPO}\n")

print("--- get_repo_info ---")
print(get_repo_info.invoke({"repo_url": TEST_REPO}))

print("\n--- list_repo_files ---")
print(list_repo_files.invoke({"repo_url": TEST_REPO, "path": ""}))

print("\n--- get_file_content (README) ---")
print(get_file_content.invoke({"repo_url": TEST_REPO, "file_path": "README"}))

print("\n✅ Якщо бачиш реальні дані вище (не помилки) - інструменти працюють правильно!")
