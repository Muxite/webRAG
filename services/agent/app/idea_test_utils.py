"""
Shared utilities for idea test system.
"""

import json
import inspect
from typing import Dict, Any, Optional


def extract_final_text(result: Dict[str, Any]) -> str:
    """
    Extract final text from test result output.
    :param result: Test result dict.
    :return: Final text string.
    """
    output = result.get("output", {})
    if not isinstance(output, dict):
        return str(output)
    
    final_deliverable = output.get("final_deliverable", "")
    if isinstance(final_deliverable, str):
        return final_deliverable
    elif isinstance(final_deliverable, dict):
        return json.dumps(final_deliverable, ensure_ascii=True)
    elif isinstance(final_deliverable, list):
        return json.dumps(final_deliverable, ensure_ascii=True)
    return str(final_deliverable)


async def call_validation_function(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Call validation function (async or sync).
    :param func: Validation function.
    :param args: Positional arguments.
    :param kwargs: Keyword arguments.
    :return: Validation result dict.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


def count_words(text: str) -> int:
    """
    Count words in text.
    :param text: Input text.
    :return: Word count.
    """
    return len(str(text).split())


def count_chars(text: str) -> int:
    """
    Count characters in text.
    :param text: Input text.
    :return: Character count.
    """
    return len(str(text))
