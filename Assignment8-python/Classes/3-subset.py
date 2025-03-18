# 3) Write a Python class to get all possible unique subsets from a set of distinct integers 
# Input : [4, 5, 6] 
# Output : [[], [6], [5], [5, 6], [4], [4, 6], [4, 5], [4, 5, 6]] 

class SubsetGenerator:
    def get_subsets(self, nums):
        result = [[]]  
        for num in nums:
            result += [subset + [num] for subset in result]  
        return result
generator = SubsetGenerator()
nums = [4, 5, 6]
print(generator.get_subsets(nums))
