import numpy as np

# Create a 50x50 grid filled with 0s
grid = np.zeros((50, 50), dtype=int)

# Define the center of the grid and the size of the diagonal square (side length of square along the diagonal)
center_x, center_y = 25, 25
side_length = 30  # Adjust this value for larger or smaller squares

# Loop over each cell in the grid
for i in range(50):
    for j in range(50):
        # Calculate the Manhattan distance from the center to see if it falls within the diamond
        if abs(i - center_x) + abs(j - center_y) < side_length / 2:
            grid[i, j] = 1

# Print the grid
for row in grid:
    print(",".join(map(str, row)))

