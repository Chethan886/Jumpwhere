# 16. Take inputs from user to make a list.
# Again take one input from user and search it in the list and delete that element, if found. Iterate over list using for loop.

nums=list(map(int,input("Enter the numbers separated by space: ").split()))
n=int(input("Enter the number to search and delete: "))
a=-1
for i in range(len(nums)):
    if nums[i]==n:
        a=i
if a!=-1:
    nums.pop(a)
else:
    print("Number not found")
print(nums)