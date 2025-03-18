# 19. From a list containing ints, strings and floats, make three lists to store them separately
# Function to separate elements based on type

def separate_types(mixed_list):
    int_list = []
    str_list = []
    float_list = []

    for item in mixed_list:
        if isinstance(item, int):
            int_list.append(item)
        elif isinstance(item, float):
            float_list.append(item)
        elif isinstance(item, str):
            str_list.append(item)

    return int_list, str_list, float_list

mixed_list = [10, "hello", 3.14, 42, "world", 99, 2.718, "python", 100.5]
integers, strings, floats = separate_types(mixed_list)
print("Integers:", integers)
print("Strings:", strings)
print("Floats:", floats)
