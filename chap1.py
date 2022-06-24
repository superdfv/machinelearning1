import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

z = np.array([[1, 2, 3],[4, 5, 6]])  #matriz
print('x:\n{}'.format(z))

eye = np.eye(4)
print('Numpy array:\n{}'.format(eye)) #matriz 4x4 diagonal 1
sparse_matrix = sparse.csr_matrix(eye)
print('\nScipy sparse CSR matrix:\n{}'.format(sparse_matrix)) # coord y value

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO representation:\n{}'.format(eye_coo)) #coord y value

#generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
#second array using sine
y = np.sin(x)
#plot
plt.plot(x, y, marker='x')
plt.show()

#using pandas

datica = {'Name': ["John", "Anna", "Peter", "Linda"],
            'Location' : ["New York", "Paris", "Berlin", "London"],
            'Age' : [24, 13, 53, 33]
}

datica_pandas = pd.DataFrame(datica)
print(datica_pandas[datica_pandas.Age > 30])