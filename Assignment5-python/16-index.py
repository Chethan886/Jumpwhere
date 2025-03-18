#16. Write a Python program to print the index of the character in a string.

def print_char_indexes(s):
    for index, char in enumerate(s):
        print(f"Character '{char}' is at index {index}")

s = input("Enter a string: ")
print_char_indexes(s)