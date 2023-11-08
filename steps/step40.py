if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F
import numpy as np

x0=Variable(np.array([1,2,3]))
x1=Variable(np.array([10]))

y=x0+x1
print(y)

y.backward()
print(x1.grad)
