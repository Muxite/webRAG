"""
Fallback logging utilities to ensure errors are logged even if primary logging fails.
"""

import sys
import traceback
from typing import Any, Optional


def safe_log_stderr(level: str, message: str, *args, **kwargs) -> None:
    """
    Log to stderr as fallback when primary logging fails.
    :param level: Log level (INFO, ERROR, etc.)
    :param message: Log message
    :param args: Additional positional args
    :param kwargs: Additional keyword args
    :returns None: Nothing is returned
    """
    try:
        formatted_msg = message
        if args:
            formatted_msg = message % args if '%' in message else f"{message} {args}"
        if kwargs:
            formatted_msg = f"{formatted_msg} {kwargs}"
        print(f"[FALLBACK {level}] {formatted_msg}", file=sys.stderr, flush=True)
    except Exception:
        try:
            print(f"[FALLBACK {level}] {str(message)}", file=sys.stderr, flush=True)
        except Exception:
            pass


def safe_log_exception(exc: Exception, context: str = "UNKNOWN") -> None:
    """
    Log exception to stderr with traceback as fallback.
    :param exc: Exception to log
    :param context: Context where exception occurred
    :returns None: Nothing is returned
    """
    try:
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        tb_str = traceback.format_exc()
        print(
            f"[FALLBACK ERROR] Exception in {context}: {exc_type}: {exc_msg}\n{tb_str}",
            file=sys.stderr,
            flush=True
        )
    except Exception:
        try:
            print(f"[FALLBACK ERROR] Exception in {context}: {str(exc)}", file=sys.stderr, flush=True)
        except Exception:
            pass
