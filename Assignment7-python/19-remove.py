# 19. Write a Python program to remove duplicates from a list of lists.
# Sample list : [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
# New List : [[10, 20], [30, 56, 25], [33], [40]]

def remove_duplicates(lst):
    unique_lists = []
    seen = set()
    
    for sublist in lst:
        tuple_sublist = tuple(sublist)  
        if tuple_sublist not in seen:
            seen.add(tuple_sublist)
            unique_lists.append(sublist) 
    
    return unique_lists

sample_list = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
new_list = remove_duplicates(sample_list)
print("New List:", new_list)
