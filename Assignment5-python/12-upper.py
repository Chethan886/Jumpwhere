# 12. Write a Python function to convert a given string to all uppercase if it contains at least 2 uppercase characters in the first 4 characters.

def upper(s):
    count = sum(1 for c in s[:4] if c.isupper())
    if count>=2:
        return s.upper()
    return s
s=input("Enter a string: ")
print(upper(s))