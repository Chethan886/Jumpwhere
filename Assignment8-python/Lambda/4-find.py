# 4) Write a Python program to find if a given string starts with a given character using Lambda.

starts_with = lambda s, char: s.startswith(char)

string = "Hello"
char = "H"
print(f"Does '{string}' start with '{char}'? {starts_with(string, char)}")
