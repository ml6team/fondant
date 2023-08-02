import re


def mainly_uppercase(line, threshold=0.7):
    """
    Checks if a line is mainly composed of uppercase characters.

    Args:
        line (str): The input line to check.
        threshold (float): The threshold (between 0 and 1) to determine what is considered
        "mainly uppercase."

    Returns:
        bool: True if the line is mainly uppercase, False otherwise.
    """
    uppercase_count = sum(1 for char in line if char.isupper())
    total_chars = len(line)
    if total_chars == 0:
        return False

    uppercase_ratio = uppercase_count / total_chars
    return uppercase_ratio >= threshold


def only_numerical(line):
    """
    Checks if a line is composed only of numerical characters.

    Args:
        line (str): The input line to check.

    Returns:
        bool: True if the line is only composed of numerical characters, False otherwise.
    """
    return line.isdigit()


def is_counter(line):
    """
    Checks if a line represents a counter (e.g., "3 likes").

    Args:
        line (str): The input line to check.

    Returns:
        bool: True if the line represents a counter, False otherwise.
    """
    # Use regular expression to check for the pattern: <number> <text>
    line = line.strip()
    pattern = r"^\d+\s+\S+$"
    return re.match(pattern, line) is not None


def is_one_word(line):
    """
    Checks if a line contains only one word.

    Args:
        line (str): The input line to check.

    Returns:
        bool: True if the line contains only one word, False otherwise.
    """
    words = line.split()
    return len(words) == 1
