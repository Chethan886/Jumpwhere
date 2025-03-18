# 15. Take integer inputs from user until he/she presses q ( Ask to press q to quit after every integer input ). 
# Print average and product of all numbers.
numbers = [] 
product = 1  

while True:
    user_input = input("Enter a number (or press 'q' to quit): ")
    if user_input.lower() == 'q': 
        break
    try:
        num = int(user_input)  # Convert input to integer
        numbers.append(num)  # Store the number
        product *= num  # Multiply for product
    except ValueError:
        print("Invalid input! Please enter an integer or 'q' to quit.")

if numbers: 
    average = sum(numbers) / len(numbers)
    print("\nAverage:", average)
    print("Product:", product)
else:
    print("\nNo numbers were entered.")
