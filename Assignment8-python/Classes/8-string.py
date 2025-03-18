# 8) Write a python class which has 2 methods get_string and print_string.
#  get_string takes a string from the user and print_string prints the string in reverse order 

class StringManipulator:
    def __init__(self):
        self.s = ""

    def get_string(self):
        self.s = input("Enter a string: ") 

    def print_string(self):
        print("Reversed String:", self.s[::-1]) 

# Example usage:
manipulator = StringManipulator()
manipulator.get_string()
manipulator.print_string()
