def shift_coordinates(coords):
    min_x = min(x for x, y in coords)
    min_y = min(y for x, y in coords)
    
    new_coords = [(x - min_x, y - min_y) for x, y in coords]
    return new_coords

coords = [(1,-2), (-2, 4), (-1,-1),(-8, -3), (0, 4), (10,-3)]

result = shift_coordinates(coords)
print(result)
