
# 18. Write a Python program to check if all dictionaries in a list are empty or not.
# Sample list : [{},{},{}]
# Return value : True
# Sample list : [{1,2},{},{}]
# Return value : False

def all_dicts_empty(lst):
    return all(not d for d in lst)  

list1 = [{},{},{}]
list2 = [{1,2},{},{}]
print("All dictionaries empty in list1:", all_dicts_empty(list1))  
print("All dictionaries empty in list2:", all_dicts_empty(list2))  
