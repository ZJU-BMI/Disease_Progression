import numpy as np
import scipy

t_y = np.array([1, 2, 3, 4, 5, 6, 7])
f_y = np.fft.fft(t_y)
print(t_y)
print(f_y)
