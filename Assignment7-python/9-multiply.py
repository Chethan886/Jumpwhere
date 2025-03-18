# 9. Write a Python program to multiply all the items in a dictionary.
def mul(d):
    result = 1
    for value in d.values():
        result *= value
    return result

sample_dict = {'a': 2, 'b': 3, 'c': 4}
print("Product of all values:", mul(sample_dict))
