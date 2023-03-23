import slidingwindow as sw
import numpy as np

dim = 10
mat = np.arange(dim**2).reshape(dim, dim)

print(mat)


windows = sw.generate(mat, sw.DimOrder.HeightWidthChannel, 6, 0.1) # < 1024, do cookie cutter, 
    
for window in windows:
	print(mat[window.indices()])