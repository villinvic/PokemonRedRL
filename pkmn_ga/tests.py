import numpy as np

# Create a sample NumPy array
array = np.array([0, 1, 2, 3, 0, 1, 0, 0, 4, 0, 5, 0, 1, 6])

# Find the indices where the pattern (0, 1) occurs
indices = np.where((array[:-1] == 0) & (array[1:] == 0))[0]

# Remove elements at the found indices
result_array = np.delete(array, np.stack([indices, indices+1]))

print("Original array:", array)
print("Array after removing indices:", result_array)