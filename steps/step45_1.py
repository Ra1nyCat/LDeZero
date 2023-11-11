if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F
import numpy as np
import dezero.layers as L
from dezero import Model

np.random.seed(0)
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

#设置超参数
lr=0.2
max_iter=10000
hidden_size=10

#定义模型
class TwoLayerNet(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.l1=L.Linear(hidden_size)
        self.l2=L.Linear(out_size)
    
    def forward(self,x):
        y=F.sigmoid(self.l1(x))
        y=self.l2(y)
        return y

model=TwoLayerNet(hidden_size,1)

#开始训练
for i in range(max_iter):
    y_pred=model(x)
    loss=F.mean_squared_error(y,y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data-=lr*p.grad.data
    
    if i%1000==0:
        print(loss)

