# 18. From the two list obtained in previous question, make new lists, 
# containing only numbers which are divisible by 4, 6, 8, 10, 3, 5, 7 and 9 in separate lists.

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

def filter_divisible(numbers, divisor):
    return [num for num in numbers if num % divisor == 0]

even_div_by_3 = filter_divisible(even, 3)
even_div_by_4 = filter_divisible(even, 4)
even_div_by_5 = filter_divisible(even, 5)
even_div_by_6 = filter_divisible(even, 6)
even_div_by_7 = filter_divisible(even, 7)
even_div_by_8 = filter_divisible(even, 8)
even_div_by_9 = filter_divisible(even, 9)
even_div_by_10 = filter_divisible(even, 10)

print("Even Numbers divisible by 3:", even_div_by_3)
print("Even Numbers divisible by 4:", even_div_by_4)
print("Even Numbers divisible by 5:", even_div_by_5)
print("Even Numbers divisible by 6:", even_div_by_6)
print("Even Numbers divisible by 7:", even_div_by_7)
print("Even Numbers divisible by 8:", even_div_by_8)
print("Even Numbers divisible by 9:", even_div_by_9)
print("Even Numbers divisible by 10:", even_div_by_10)

div_by_3 = filter_divisible(odd, 3)
div_by_5 = filter_divisible(odd, 5)
div_by_7 = filter_divisible(odd, 7)
div_by_9 = filter_divisible(odd, 9)
print("Odd Numbers divisible by 3:", div_by_3)
print("Odd Numbers divisible by 5:", div_by_5)
print("Odd Numbers divisible by 7:", div_by_7)
print("Odd Numbers divisible by 9:", div_by_9)

prime_div_by_3 = filter_divisible(prime, 3)
prime_div_by_4 = filter_divisible(prime, 4)
prime_div_by_5 = filter_divisible(prime, 5)
prime_div_by_6 = filter_divisible(prime, 6)
prime_div_by_7 = filter_divisible(prime, 7)
prime_div_by_8 = filter_divisible(prime, 8)
prime_div_by_9 = filter_divisible(prime, 9)
prime_div_by_10 = filter_divisible(prime, 10)

print("Prime Numbers divisible by 3:", prime_div_by_3)
print("Prime Numbers divisible by 4:", prime_div_by_4)
print("Prime Numbers divisible by 5:", prime_div_by_5)
print("Prime Numbers divisible by 6:",prime_div_by_6)
print("Prime Numbers divisible by 7:", prime_div_by_7)
print("Prime Numbers divisible by 8:", prime_div_by_8)
print("Prime Numbers divisible by 9:", prime_div_by_9)
print("Prime Numbers divisible by 10:", prime_div_by_10)