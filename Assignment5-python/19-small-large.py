# 19. Write a Python program to find smallest and largest word in a given string.

def find_large_small(words):
    min_len = 1000
    max_len = 0
    for word in words:
        if len(word) < min_len:
            min_len = len(word)
            min_word = word
        if len(word) > max_len:
            max_len = len(word)
            max_word = word
    return min_word, max_word
words = input("Enter words separated by spaces: ").split()
min_word, max_word = find_large_small(words)
print("Smallest word is:", min_word)
print("Largest word is:", max_word)