
# 7. Write a Python program that counts the number of elements within a list that are greater than 30.

nums=list(map(int,input("Enter the numbers with space : ").split()))
count=sum(1 for num in nums if num>30)
print(f"Count of numbers greater than 30: {count}")
