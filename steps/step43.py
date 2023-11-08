if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F
import numpy as np

np.random.seed(0)
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

#1.初始化参数
I,H,O=1,10,1
w1=Variable(0.01*np.random.randn(I,H))
b1=Variable(np.zeros(H))
w2=Variable(0.01*np.random.randn(H,O))
b2=Variable(np.zeros(O))

#2.定义模型
def predict(x):
    y=F.linear(x,w1,b1)
    y=F.sigmoid(y)
    y=F.linear(y,w2,b2)
    return y


lr=0.2
iters=10000

#3.训练模型
for i in range(iters):
    y_pred=predict(x)
    loss=F.mean_squared_error(y,y_pred)

    w1.cleargrad()
    b1.cleargrad()
    w2.cleargrad()
    b2.cleargrad()
    loss.backward()

    w1.data-=lr*w1.grad.data
    b1.data-=lr*b1.grad.data
    w2.data-=lr*w2.grad.data
    b2.data-=lr*b2.grad.data
    
    if i%1000==0:
        print(loss)