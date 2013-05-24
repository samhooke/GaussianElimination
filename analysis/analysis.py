'''
Outputs a simple analysis of data dapendencies for Gaussian Elimination. The
algorithm can be split into stages defined by the pivot position. The pivot
position directly affects which elements are required by other elements to
calculate their new values.

A rectangular area to be calculated is defined by `tile`. The matrix size is
defined by `size`. A `size` of x generates a matrix of height x, and of width
x + 1.

The outputted matrix indicates which elements the `tile` relies upon during
which stage of the algorithm. Each number represents what the pivot value would
be for that particular position of dependency. An element that consists
entirely of dots indicates no dependency.
'''

import sys
import os

# Choose the matrix size
size = 8;

# Choose tile x, y, width, height
tile = [2, 2, 4, 1];

# Choose for which pivots between 0 and size - 1 to display results
stages = range(size)

reference = [(x, y) for x in range(tile[0], tile[0] + tile[2])
					for y in range(tile[1], tile[1] + tile[3])]
height = size;
width = height + 1;
matrix = {(x,y):"" for x in range(width) for y in range(height)}

for pivot in range(height):
	if pivot in stages:
		for r in reference:
			matrix[(pivot, r[1])] += str(pivot + 1)
			matrix[(r[0], pivot)] += str(pivot + 1)

os.system(['clear', 'cls'][os.name == 'nt'])

for y in range(height):
	for x in range(width):
		s = matrix[(x,y)]
		while (len(s) < len(reference) + 1):
			s += "."
		s += " "
		sys.stdout.write(s)
	sys.stdout.write('\n')
