# 16. Write a Python program to find the highest 3 values in a dictionary.
def top_three_values(d):
    return sorted(d.values(), reverse=True)[:3]

sample_dict = {'a': 10, 'b': 25, 'c': 5, 'd': 40, 'e': 30}
print("Top 3 highest values:", top_three_values(sample_dict))
