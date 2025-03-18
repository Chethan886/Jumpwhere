# 20.You are given with a list of integer elements. Make a new list which will store square of elements of previous list

def square_list(nums):
    return [x ** 2 for x in nums]  

nums = [2, 4, 6, 8, 10]

squared_list = square_list(nums)

print("Original List:", nums)
print("Squared List:", squared_list)
