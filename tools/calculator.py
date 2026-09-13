"""
Ultron - calculator tool.

Evaluates simple mathematical expressions safely.
"""
import re
import ast
import operator
from utils.logger import get_logger

log = get_logger("tools.calculator")

# Supported operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in _OPERATORS:
            return _OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Expression):
        return _eval_node(node.body)
    else:
        raise ValueError(f"Unsupported node type: {type(node)}")


def evaluate_expression(expr: str):
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expr: Mathematical expression string (e.g., "2 + 2", "10 * 5", "(3 + 4) * 2")
    
    Returns:
        Result as float/int or error message string
    """
    expr = expr.strip()
    if not expr:
        return "Empty expression"
    
    # Basic validation - only allow numbers, operators, parentheses, and whitespace
    if not re.match(r'^[\d\s+\-*/().%]+$', expr):
        return "Invalid characters in expression"
    
    try:
        # Parse the expression
        tree = ast.parse(expr, mode='eval')
        result = _eval_node(tree.body)
        # Return int if whole number, else float
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
    except ZeroDivisionError:
        return "Division by zero"
    except Exception as e:
        log.debug("Calculator error: %s", e)
        return f"Invalid expression: {e}"


def detect_calculation(text: str) -> bool:
    """Check if the text looks like a calculation request."""
    # Match patterns like "2 + 2", "calculate 10 * 5", "what is 3 * 4"
    patterns = [
        r'^\s*[\d\s+\-*/().%]+\s*$',  # Pure expression
        r'\b(?:calculate|compute|eval)\b.*[\d\s+\-*/().%]{3,}',  # "calculate 2+2" or "compute 10 * 5"
        r"\bwhat\s+(?:is|'s)\b.*[\d\s+\-*/().%]{3,}",  # "what is 3 * 4"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def extract_expression(text: str) -> str:
    """Extract the mathematical expression from text."""
    # Try to find a pure expression
    match = re.search(r'([\d\s+\-*/().%]{3,})', text)
    if match:
        expr = match.group(1).strip()
        # Clean up common words that might be captured
        expr = re.sub(r'\b(?:calculate|compute|eval|what|is|is|what\s+is|what\s)\b', '', expr, flags=re.IGNORECASE).strip()
        return expr
    return text.strip()


def calculator_answer(query: str) -> str:
    """Process a calculation query and return formatted answer."""
    expr = extract_expression(query)
    result = evaluate_expression(expr)
    if isinstance(result, str) and result.startswith(("Invalid", "Division", "Empty")):
        return f"Calculator error: {result}"
    return f"**Calculator**  \nExpression: `{expr}`  \nResult: **{result}**"