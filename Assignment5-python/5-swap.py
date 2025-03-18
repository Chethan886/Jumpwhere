# 5. Write a Python program to get a single string from two given strings, separated by a space and swap the first two characters of each string.
# Sample String : 'abc', 'xyz' 
# Expected Result : 'xyc abz'


def swap(s1,s2):
    if len(s1)<2 and len(s2)<2:
        return 'length of the string is less than 2'
    swap1=s2[:2]+s1[2:]
    swap2=s1[:2]+s2[2:]
    return swap1+' '+swap2
s1=input('Enter the first string: ')
s2=input('Enter the second string: ')
print(swap(s1,s2))