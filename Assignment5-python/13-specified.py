#13. Write a Python program to check whether a string starts with specified characters.

def specified(ch,s):
    if s[0] == ch:
        print("The string starts with the specified character")
    else:
        print("The string does not start with the specified character")
s=input("Enter a string: ")
ch=input("Enter a character: ")
specified(ch,s)
