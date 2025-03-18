# 4. Write a Python program to get a string from a given string where all occurrences of its first char have been changed to '$', except the first char itself.
# Sample String : 'restart'
# Expected Result : 'resta$t'


def change(s):
    a=s[0]
    str1=a+s[1:].replace(a,'$')
    return str1
s=input("Enter a string: ")
print(change(s))