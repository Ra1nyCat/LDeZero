import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self,func):
        self.creator=func

    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:
            f=funcs.pop() #取出一个函数
            x,y=f.input,f.output
            x.grad=f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self,input:Variable)->Variable:
        x=input.data
        y=self.forward(x)
        output=Variable(self._as_array(y))
        output.set_creator(self)    #设置创造者
        self.input=input #保存输入的变量
        self.output=output #保存输出的变量
        return output
    
    def _as_array(self,x): #将输入转换为numpy数组
        if np.isscalar(x): #判断是否为标量
            return np.array(x)
        return x
    
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



x=Variable(np.array(0.5))
y=square(exp(square(x)))
y.backward()
print(x.grad)