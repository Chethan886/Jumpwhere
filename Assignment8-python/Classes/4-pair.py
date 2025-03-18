# 4) Write a Python class to find a pair of elements (indices of the two numbers) from a given array whose sum equals a specific target number. Note: There will be one solution for each input and do not use the same element twice. 


class PairFinder:
    def find_pair(self, numbers, target):
        num_map = {}  
        for i, num in enumerate(numbers):
            complement = target - num
            if complement in num_map:
                return num_map[complement], i  
            num_map[num] = i  
        
        return None  

numbers = [90, 20, 10, 40, 50, 60, 70]
target = 50
finder = PairFinder()
print(finder.find_pair(numbers, target))
