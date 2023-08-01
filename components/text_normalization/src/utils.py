import re


def mainly_uppercase(line, threshold=0.7):
    """
    Checks if a line is mainly composed of uppercase characters.

    Args:
        line (str): The input line to check.
        threshold (float): The threshold (between 0 and 1) to determine what is considered "mainly uppercase."

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


def read_patterns_from_file(file_path):
    """
    Read patterns from a text file.

    Args:
        file_path (str): The path to the text file containing patterns.

    Returns:
        list: A list of patterns read from the file.
    """
    with open(file_path) as file:
        return [pattern.strip() for pattern in file]
def is_short_and_matches_pattern(line, pattern_file_path, max_words=10):
    """
    Checks if a line is short (< max_words) and matches the given pattern.

    Args:
        line (str): The input line to check.
        max_words (int): The maximum number of words allowed in the line (default is 10).

    Returns:
        bool: True if the line is short and matches the pattern, False otherwise.
    """
    patterns = read_patterns_from_file(pattern_file_path)
    words = line.split()
    if len(words) > max_words:
        return False

    for pattern in patterns:
        if re.search(rf'\b{re.escape(pattern)}\b', line) is not None:
            return True
    return None

