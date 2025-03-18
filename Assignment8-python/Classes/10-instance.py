# 10) Write a Python program to get the class name of an instance in Python.

class SampleClass:
    pass 

# Create an instance
obj = SampleClass()

print(obj.__class__.__name__)  
