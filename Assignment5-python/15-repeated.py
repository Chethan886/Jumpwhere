
# 15. Write a Python program to count repeated characters in a string.
# Sample string: 'thequickbrownfoxjumpsoverthelazydog'
# Expected output :
# o 4
# e 3
# u 2
# h 2
# r 2
# t 2


def repeated(s):
    freq={}
    for i in s:
        freq[i]=freq.get(i,0)+1
    repeated_chars = {char: count for char, count in freq.items() if count > 1}

    return repeated_chars
s=input("Enter a string:")
print(repeated(s))