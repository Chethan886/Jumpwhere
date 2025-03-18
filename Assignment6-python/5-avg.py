# 5. Write a Python program to calculate the sum and average of n integer numbers (input from the user). Input 0 to finish

count=0
total=0
while True:
    num=int(input("Enter a number (0 to finish): "))
    if num==0:
        break
    count+=1
    total+=num
if count==0:
    print('No numbers entered')
else:
    print("Sum",total)
    print('Average:',total/count)
