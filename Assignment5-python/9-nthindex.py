#9. Write a Python program to remove the nth index character from a nonempty string.

def remove_nth_index(string, n):
    return string[:n] + string[n+1:]
string = input("Enter a string: ")
n = int(input("Enter the index to remove: "))
print(remove_nth_index(string, n))