# 10. Write a Python program that accepts a comma separated sequence of words as input and prints the unique words in sorted form (alphanumerically). 
# Sample Words : red, white, black, red, green, black
# Expected Result : black, green, red, white


def wordsort(words):
    words.sort()
    return words
words = input("Enter words separated by commas: ").split(',')
print(wordsort(words))