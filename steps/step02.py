from step01 import Variable
import numpy as np

class Function:
    def __call__(self,input:Variable)->Variable:
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        output.set_creator(self)    #设置创造者
        self.input=input #保存输入的变量
        self.output=output #保存输出的变量
        return output
    
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self,gy): #gy是输出方向的导数
        raise NotImplementedError()
    

class Square(Function):
    def forward(self,x):
        return x**2

    def backward(self,gy): 
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def numerical_diff(f:Function,x:Variable,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)


def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1.0)
y.backward()
print(x.grad)