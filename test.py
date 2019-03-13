import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])

res = np.stack((a,b,c),axis=2)
print(res.shape)
print(res)