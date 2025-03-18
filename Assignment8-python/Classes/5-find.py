# 5) Write a Python class to find the three elements that sum to zero from a set of n real numbers. 
# Input array : [-25, -10, -7, -3, 2, 4, 8, 10]
#  Output : [[-10, 2, 8], [-7, -3, 10]] 

class ThreeSumFinder:
    def three_sum(self, nums):
        nums.sort()  
        result = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:  
                continue            
            left, right = i + 1, len(nums) - 1           
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]: 
                        left += 1
                    while left < right and nums[right] == nums[right + 1]: 
                        right -= 1
                elif total < 0:
                    left += 1 
                else:
                    right -= 1     
        return result

finder = ThreeSumFinder()
nums = [-25, -10, -7, -3, 2, 4, 8, 10]
print(finder.three_sum(nums))
