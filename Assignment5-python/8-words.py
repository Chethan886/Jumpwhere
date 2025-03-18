#8. Write a Python function that takes a list of words and returns the length of the longest one. 

def wordslist(words):
    maxlen=-1
    for word in words:
        if len(word)>maxlen:
            maxlen=len(word)
            
    return maxlen
words = input("Enter words separated by spaces: ").split()  
print(wordslist(words))
