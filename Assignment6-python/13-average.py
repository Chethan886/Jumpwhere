# 13. Take 10 integers from keyboard using loop and print their average value on the screen.
sum=0
for i in range(10):
    n=int(input("Enter a number: "))
    sum+=n
print("Average:",sum/10)