# assistant/tools.py
"""Minimal tool use: a safe calculator and a date/time tool, with a router."""
import ast
import operator
import re
from datetime import datetime

# Safe arithmetic: only allow math operators, never arbitrary code.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return f"The answer is {_safe_eval(tree.body)}."
    except Exception:
        return "I couldn't evaluate that expression."


def current_datetime() -> str:
    now = datetime.now()
    return f"The current date and time is {now.strftime('%A, %d %B %Y, %H:%M')}."


def maybe_use_tool(text: str) -> str | None:
    """Return a tool result if the input clearly calls for one, else None."""
    lowered = text.lower()

    if any(w in lowered for w in ["what time", "what's the date", "today's date", "current date", "what day"]):
        return current_datetime()

    # A pure-ish arithmetic query, e.g. "what is 23 * 47" or "12 + 5 / 2"
    math_match = re.search(r"[-+]?\d[\d\s+\-*/().%^]*\d", text)
    if math_match and any(op in text for op in "+-*/^%"):
        expr = math_match.group().replace("^", "**")
        return calculator(expr)

    return None