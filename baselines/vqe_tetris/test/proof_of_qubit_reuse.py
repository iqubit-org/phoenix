import numpy as np
from toqito.random import random_unitary
import ipdb

# q0, q1

I = np.array([[1.0, 0.], [0., 1.0]])
q0 = np.array([[1.0, 0.], [0., 0.]])
q1 = np.array([[1.0, 0.], [0., 0.]])
density = np.kron(q0, q1)
print(density)

mat = random_unitary([4, 4], False)
mat2 = random_unitary([2, 2], False)

# ipdb.set_trace()
density = np.matmul(np.matmul(mat, density), mat.conj().T)

# measure q0 before operations on q1
density2 = density.reshape([2, 2, 2, 2])
density2 = np.trace(density2, axis1=0, axis2=2)
density2 = np.matmul(density2, mat2)
print(density2)

# do operations on q1 before measure q0
density3 = np.matmul(density, np.kron(I, mat2))
density3 = density3.reshape([2, 2, 2, 2])
density3 = np.trace(density3, axis1=0, axis2=2)
print(density3)
