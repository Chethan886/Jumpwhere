# 18. Write a Python program to swap comma and dot in a string.
# Sample string: "32.054,23"
# Expected Output: "32,054.23"

def swap(s):
    return s.replace(',', 'temp').replace('.', ',').replace('temp', '.')

s=input("Enter a string: ")
print(swap(s))