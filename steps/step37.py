if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F
import numpy as np

x=Variable(np.random.randn(1,2,3))
print(x)

x=Variable(np.array([[1,2,3],[4,5,6]]))
y=F.transpose(x)
y.backward()
print(x.grad)
