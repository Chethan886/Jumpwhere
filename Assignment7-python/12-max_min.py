# 12. Write a Python program to get the maximum and minimum value in a dictionary.

def get_max_min_values(d):
    if not d:
        return None, None 
    max_value = max(d.values())
    min_value = min(d.values())
    return max_value, min_value

sample_dict = {'a': 10, 'b': 25, 'c': 5, 'd': 40}

max_val, min_val = get_max_min_values(sample_dict)
print("Maximum value:", max_val)
print("Minimum value:", min_val)
