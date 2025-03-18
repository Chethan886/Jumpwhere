# 9) Write a Python program to find the elements of a given list of strings that contain specific substring using lambda. 
# Original list: ['red', 'black', 'white', 'green', 'orange'] 
# Substring to search: ack Elements of the said list that contain specific substring: ['black'] Substring to search: abc Elements of the said list that contain specific substring: [] 

find_substring = lambda words, sub: list(filter(lambda word: sub in word, words))
words_list = ['red', 'black', 'white', 'green', 'orange']

substring1 = "ack"
result1 = find_substring(words_list, substring1)
print(f"Elements that contain '{substring1}':", result1)

substring2 = "abc"
result2 = find_substring(words_list, substring2)
print(f"Elements that contain '{substring2}':", result2)
