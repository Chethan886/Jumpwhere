# 6. Write a Python program to construct the following pattern, using a nested loop number.
# 1
# 22
# 333
# 4444
# 55555
# 666666
# 7777777
# 88888888
# 999999999


c=1
for i in range(9):
    for j in range(c):
        print(c,end="")
    print()
    c+=1
