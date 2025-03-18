# 13. Write a Python program to remove duplicates from Dictionary.
def remove_duplicates(d):
    values = {}
    for key, value in d.items():
        if value not in values.values():
            values[key] = value
    return values

sample_dict = {'a': 10, 'b': 20, 'c': 10, 'd': 30, 'e': 20}
unique_dict = remove_duplicates(sample_dict)
print("Dictionary after removing duplicates:", unique_dict)
