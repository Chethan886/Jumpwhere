#20. Write a Python program to remove all consecutive duplicates of a given string.

def duplicate(s):
    res=s[0]
    for i in range(1,len(s)):
        if s[i]!=s[i-1]:
            res+=s[i]

    return res
s=input("Enter a string: ")
print(duplicate(s))