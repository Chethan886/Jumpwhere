#11. Write a Python function to reverses a string if it's length is a multiple of 4. 

def reverse_string(string):
    if len(string)%4 == 0:
        return string[::-1]
    return string
string = input("Enter a string: ")
print(reverse_string(string))