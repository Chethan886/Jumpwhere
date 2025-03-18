# 7. Write a Python program to find the first appearance of the substring 'not' and 'poor' from a given string, if 'not' follows the 'poor', replace the whole 'not'...'poor' substring with 'good'. Return the resulting string.
# Sample String : 'The lyrics is not that poor!'
# 'The lyrics is poor!'
# Expected Result : 'The lyrics is good!'
# 'The lyrics is poor!'


def poor(s):
    not_i=s.find('not')
    poor_i=s.find('poor')
    if not_i!=-1 and poor_i!=-1 and poor_i>not_i:
        s=s.replace(s[not_i:poor_i+4],'good')
    return s
s=input("Enter a string: ")
print(poor(s))