if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F
import numpy as np

np.random.seed(0)
x=np.random.rand(100,1)
y=5+2*x+np.random.rand(100,1)

x,y=Variable(x),Variable(y)
w=Variable(np.random.rand(1,1))
b=Variable(np.random.rand(1))

def predict(x):
    y=F.matmul(x,w)+b
    return y


lr=0.1
iters=100

for i in range(iters):
    y_pred=predict(x)
    loss=F.mean_squared_error(y,y_pred)

    w.cleargrad()
    b.cleargrad()
    loss.backward()

    w.data-=lr*w.grad.data
    b.data-=lr*b.grad.data
    print(w,b,loss)