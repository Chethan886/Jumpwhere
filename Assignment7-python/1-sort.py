# 1. Write a Python script to sort (ascending and descending) a dictionary by value.

my_dict = {'apple': 50, 'banana': 30, 'cherry': 40, 'date': 60}

asc = dict(sorted(my_dict.items(), key=lambda item: item[1]))

desc = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
print("Ascending Order:", asc)
print("Descending Order:", desc)
