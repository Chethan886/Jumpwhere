
# 10. A company decided to give bonus of 5% to employee if his/her year of service is more than 5 years.
# Ask user for their salary and year of service and print the net bonus amount.

sal=int(input('Enter your salary: '))
years=int(input('Enter your total years of work: '))
if years>5:
    bonus=sal*0.05
    sal=sal+bonus
    print('salary after bonus:',sal)
else:
    print("No bonus")