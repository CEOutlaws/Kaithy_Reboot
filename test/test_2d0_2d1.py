import numpy as np
a = np.array([[1, 2], [3, 4]])
print(a)
print(a.shape)
a = np.stack((a,) * 1, axis=-1)
print(a)
print(a.shape)
print(a[0][0][0])
