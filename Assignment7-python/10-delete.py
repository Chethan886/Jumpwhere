# 10. Write a Python program to remove a key from a dictionary. 

def remove_key(d, key):
    if key in d:
        del d[key]
    return d

dict = {'a': 1, 'b': 2, 'c': 3}
key = 'b'

updated_dict = remove_key(dict, key)
print("Updated dictionary:", updated_dict)
