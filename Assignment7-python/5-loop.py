# 5. Write a Python program to iterate over dictionaries using for loops.

dict = {1:"Apple",2:"Banana",3:"Cherry"}
print("Keys:")
for key in dict:
    print(key)

print("\nValues:")
for value in dict.values():
    print(value)

print("\nKey-Value Pairs:")
for key, value in dict.items():
    print(f"{key}: {value}")
