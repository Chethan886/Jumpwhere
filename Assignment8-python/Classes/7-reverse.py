# 7) Write a Python class to reverse a string word by word. 
# Input string : 'hello .py' Expected Output : '.py hello' 

class StringReverser:
    def reverse_words(self, s):
        return ' '.join(s.split()[::-1])  
reverser = StringReverser()
input_string = 'hello .py'
print(reverser.reverse_words(input_string))  
