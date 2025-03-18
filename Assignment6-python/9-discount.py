# 9. A shop will give discount of 10% if the cost of purchased quantity is more than 1000.
# Ask user for quantity
# Suppose, one unit will cost 100.
# Judge and print total cost for user.


quantity=int(input("Enter the quantity: "))
unit_price=100
cost=quantity*unit_price

if cost>1000:
    discount=cost*0.1
    cost-=discount
print('Cost:',cost)