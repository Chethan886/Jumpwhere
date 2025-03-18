# 12. A student will not be allowed to sit in exam if his/her attendence is less than 75%.
# Take following input from user
# Number of classes held
# Number of classes attended.
# And print
# percentage of class attended
# Is student is allowed to sit in exam or not.


total=int(input("Enter the total number of classes held: "))
attended=int(input("Enter the number of classes attended: "))
percentage=(attended/total)*100
print('Percentage of class attended:',percentage)
if percentage<75:
    print('Student is not allowed to sit in exam')
else:
    print('Student is allowed to sit for exam')