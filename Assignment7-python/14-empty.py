# 14. Write a Python program to check a dictionary is empty or not. 

def is_dict_empty(d):
    return not bool(d)  

empty_dict = {}
non_empty_dict = {'a': 1, 'b': 2}
if is_dict_empty(empty_dict):
    print("The dict is empty", ) 
else:
    print("The dict is not  empty") 
if is_dict_empty(non_empty_dict):
    print("The dict is empty", ) 
else:
    print("The dict is not empty") 