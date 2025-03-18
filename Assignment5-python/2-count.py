#2. Write a Python program to count the number of characters (character frequency) in a string. 
#Sample String : google.com'
#Expected Result : {'o': 3, 'g': 2, '.': 1, 'e': 1, 'l': 1, 'm': 1, 'c': 1}

def frequency(s):
    freq={}
    for i in s:
        freq[i]=freq.get(i,0)+1
    return freq
s=input("Enter a string:")
print(frequency(s))