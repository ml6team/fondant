
from src.utils import (
    is_counter,
    is_one_word,
    mainly_uppercase,
    only_numerical,
)


def test_mainly_uppercase():
    line = "HELLO WORLD not upper SOMETHING ELSE IN UPPERCASE"
    assert mainly_uppercase(line, threshold=0.5)

def test_mainly_uppercase_under_threshold():
    line = "HELLO WORLD not upper SOMETHING ELSE IN UPPERCASE"
    assert not mainly_uppercase(line, threshold=0.9)

def test_only_numerical():
    line = "42"
    assert only_numerical(line)

def test_only_numerical_on_words():
    line = "42 lorem ipsum"
    assert not only_numerical(line)

def test_is_counter():
    line = "13 Likes"
    assert is_counter(line)

def test_is_not_counter():
    line = "Hello world! 42 people are part of .."
    assert not is_counter(line)

def test_is_one_word():
    line = "word"
    assert is_one_word(line)

def test_is_not_one_word():
    line = "two words"
    assert not is_one_word(line)


