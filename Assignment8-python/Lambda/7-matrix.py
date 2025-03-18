# 7) Write a Python program to sort a given matrix in ascending order according to the sum of its rows using lambda. 
# Original Matrix: [[1, 2, 3], [2, 4, 5], [1, 1, 1]] 
# Sort the said matrix in ascending order according to the sum of its rows [[1, 1, 1], [1, 2, 3], [2, 4, 5]]
#  Original Matrix: [[1, 2, 3], [-2, 4, -5], [1, -1, 1]] 
# Sort the said matrix in ascending order according to the sum of its rows [[-2, 4, -5], [1, -1, 1], [1, 2, 3]] 

def sort_matrix_by_row_sum(matrix):
    return sorted(matrix, key=lambda row: sum(row))

matrix1 = [[1, 2, 3], [2, 4, 5], [1, 1, 1]]
matrix2 = [[1, 2, 3], [-2, 4, -5], [1, -1, 1]]

sorted_matrix1 = sort_matrix_by_row_sum(matrix1)
sorted_matrix2 = sort_matrix_by_row_sum(matrix2)

print("Sorted Matrix 1:", sorted_matrix1)
print("Sorted Matrix 2:", sorted_matrix2)
