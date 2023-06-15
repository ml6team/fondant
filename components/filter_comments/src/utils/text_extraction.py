"""This code is adapted from BigScience PII detection
https://github.com/bigcode-project/bigcode-dataset/blob/main/preprocessing/filtering.py.

MST BigScience PII Code
Original colab that is a source of this file is located at
    https://colab.research.google.com/drive/1086H3-LGMz3gX0pGy9ECgr8KflosSKso
# License
Copyright 2022 Authors of this Notebook
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ast
import io
import tokenize
import typing as t
import warnings
from itertools import groupby

NODE_TYPES = {
    ast.ClassDef: "Class",
    ast.FunctionDef: "Function/Method",
    ast.Module: "Module",
}


# Note: sometimes this can miss examples with decorators over classes
# ast parsing, source: https://gist.github.com/SpotlightKid/1548cb6c97f2a844f72d
def parse_docstrings(source):
    """
    Parse Python source code and yield a tuple of ast node instance, name,
    and docstring for each function/method, class and module.

    Args:
        source: The Python source code to parse.

    Yields:
        A tuple containing ast node instance, name, and docstring.
    """
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, tuple(NODE_TYPES)):
            docstring = ast.get_docstring(node)

            yield node, getattr(node, "name", None), docstring


# comment extraction
def get_comments(source: str) -> str:
    """
    Returns a string including all comments in python code.

    Args:
        source: the code to parse

    Returns:
        The script comments.
    """
    comments = []
    g = tokenize.generate_tokens(io.StringIO(source).readline)
    for toknum, tokval, _, _, _ in g:
        if toknum == tokenize.COMMENT:
            comments.append((toknum, tokval))
    result = tokenize.untokenize(comments).replace("#", "")

    return result


def get_docstrings(source: str) -> t.List[str]:
    """
    Parse Python source code from file or string and print docstrings.

    Args:
        source: the code to parse

    Returns:
        A list containing the script docstrings.
    """
    if hasattr(source, "read"):
        source = source.read()

    docstrings = sorted(
        parse_docstrings(source), key=lambda x: (NODE_TYPES.get(type(x[0])), x[1])
    )

    grouped = groupby(docstrings, key=lambda x: NODE_TYPES.get(type(x[0])))
    results = []
    for _, group in grouped:
        for _, name, docstring in group:
            if docstring:
                results.append(docstring)
    return results


def get_text_python(source: str, extract_comments: bool = True) -> str:
    """Extract all natural text in source: comments + docstrings
    the extraction fails in case of syntax errors in the file.

    Args:
        source: the code to parse
        extract_comments: if True extract comments too

    Returns:
        A string with concatenated docstrings and comments.
    """
    try:
        docstrings = "\n".join(get_docstrings(source))
    except Exception:
        docstrings = ""
        warnings.warn(
            "code couldn't be parsed due to compilation failure, no docstring is extracted"
        )

    if extract_comments:
        try:
            comments = get_comments(source)
        except Exception:
            comments = ""
            warnings.warn("tokenization error, no comments are extracted")
    else:
        comments = ""

    output = docstrings + "\n" + comments
    return output.strip()


def get_comments_to_code_ratio(text: str) -> float:
    """
    Get the ratio of comments to code in a program
    Args:
        text: the string source code
    Returns:
        The comments to code ratio.
    """
    comments = get_text_python(text)

    return len(comments) / len(text)