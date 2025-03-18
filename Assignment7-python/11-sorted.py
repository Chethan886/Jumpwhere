# 11. Write a Python program to sort a dictionary by key.

def sort_dict(d):
    return dict(sorted(d.items()))

sample_dict = {'b': 10, 'a': 101, 'c': 35}
sorted_dict = sort_dict(sample_dict)
print("Sorted dictionary by key:", sorted_dict)
