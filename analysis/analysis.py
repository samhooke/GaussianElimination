import sys
import os

# Choose the matrix size
size = 8;

# Choose reference positions
#reference = [(2,1), (2,2), (3,1), (3,2)]
reference = [(x, y) for x in range(2,6) for y in range(2,6)]
#reference = [(x, y) for x in range(2,4) for y in range(1,3)]

# Choose which pivots to display
pivot = range(size)

height = size;
width = height + 1;
matrix = {(x,y):"" for x in range(width) for y in range(height)}

for p in range(height):
	if p in pivot:
		px = py = p;
		for r in reference:
			matrix[(px, r[1])] += str(p+1)
			matrix[(r[0], py)] += str(p+1)

os.system(['clear', 'cls'][os.name == 'nt'])

for y in range(height):
	for x in range(width):
		s = matrix[(x,y)]
		while (len(s) < len(reference)):
			s += "."
		s += " "
		sys.stdout.write(s)
	sys.stdout.write('\n')
