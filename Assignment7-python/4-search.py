# 4. Write a Python script to check if a given key already exists in a dictionary. 

sample_dict = {1: 10, 2: 20, 3: 30, 4: 40}


def search(dictionary, key):
    if key in dictionary:
        print(f"Key {key} exists in the dictionary.")
    else:
        print(f"Key {key} does not exist in the dictionary.")

key = int(input("Enter the key to search: "))
search(sample_dict, key)
