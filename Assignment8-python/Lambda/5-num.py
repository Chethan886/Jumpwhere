# 5) Write a Python program to check whether a given string is number or not using Lambda. 

is_number = lambda s: s.replace('.', '', 1).isdigit() if s else False
test_strings = ["123", "45.67", "abc", "12e3", ""]
for s in test_strings:
    print(f"Is '{s}' a number? {is_number(s)}")
