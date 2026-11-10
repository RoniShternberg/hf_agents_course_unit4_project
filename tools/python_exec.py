"""Python code execution tool.

Uses LangChain's PythonREPL. This is not a true sandbox — the host environment
is trusted and runs on the user's machine or HF Space. We limit output size and
preserve state across calls so the agent can build up computations.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

_repl = PythonREPL()
_MAX_CHARS = 6000


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return stdout.

    State (imports, variables) persists across calls within a session. Use
    ``print()`` to surface values — the return value of the last expression
    is NOT auto-printed. Libraries available: pandas, numpy, requests,
    standard library, plus any imported at runtime.

    Example:
        python_repl("import math; print(math.sqrt(144))")
    """
    try:
        result = _repl.run(code)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
    if not isinstance(result, str):
        result = str(result)
    if len(result) > _MAX_CHARS:
        return result[:_MAX_CHARS] + f"\n…[truncated, full length {len(result)} chars]"
    return result or "[no output]"
