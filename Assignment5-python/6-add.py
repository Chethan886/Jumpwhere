# 6. Write a Python program to add 'ing' at the end of a given string (length should be at least 3). If the given string already ends with 'ing' then add 'ly' instead. If the string length of the given string is less than 3, leave it unchanged.
# Sample String : 'abc'
# Expected Result : 'abcing' 
# Sample String : 'string'
# Expected Result : 'stringly'


def add(s):
    if len(s)<3:
        return 'length should be at least 3'
    if s[-3:]=='ing':
        return s+'ly'
    return s+'ing'
s=input('Enter a string: ')
print(add(s))
    