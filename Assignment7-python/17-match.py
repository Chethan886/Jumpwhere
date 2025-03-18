# 17. Write a Python program to match key values in two dictionaries.
# Sample dictionary: {'key1': 1, 'key2': 3, 'key3': 2}, {'key1': 1, 'key2': 2}
# Expected output: key1: 1 is present in both x and y

def match_key_values(d1, d2):
    for key in d1:
        if key in d2 and d1[key] == d2[key]:
            print(f"{key}: {d1[key]} is present in both dictionaries")

dict1 = {'key1': 1, 'key2': 3, 'key3': 2}
dict2 = {'key1': 1, 'key2': 2}
match_key_values(dict1, dict2)
