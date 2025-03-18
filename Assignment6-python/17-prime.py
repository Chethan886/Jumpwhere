# 17. Using range(1,101), make three list, 
# one containing all even numbers
# one containing all odd numbers 
# One containing only prime numbers..

def is_prime(num):
    if num<2:
        return False
    for i in range(2,int(num**0.5)+1):
        if num%i==0:
            return False
    return True

even=[]
odd=[]
prime=[]
for i in range(1,101):
    if i%2==0:
        even.append(i)
    else:
        odd.append(i)
    if is_prime(i):
        prime.append(i)

print('Even:',even)
print('odd:',odd)
print('Prime:',prime)
