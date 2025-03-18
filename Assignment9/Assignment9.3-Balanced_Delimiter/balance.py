def is_balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack.pop() != pairs[char]:
                return False

    return not stack  # Stack should be empty if balanced

# Example usage
s = input("Enter a parenthesis string: ")
print(is_balanced(s))
