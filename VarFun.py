import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def cleargrad(self):
        self.grad=None

    def set_creator(self,func):
        self.creator=func
        self.generation=func.generation+1

    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_set=set()

        def add_func(f):
            if f not in seen_set: #很重要，避免重复添加
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)

        add_func(self.creator)

        while funcs:
            f=funcs.pop() #取出一个函数
            gys=[output.grad for output in f.outputs]
            gxs=f.backward(*gys) #计算出反向传播值
            if not isinstance(gxs,tuple):
                gxs=(gxs,)

            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                    print('val:{} grad-{} id-{}'.format(x.data,x.grad,id(x.data)))
                else:
                    x.grad=x.grad+gx #累加梯度,且不可写成+=，+=是in-place，会覆盖id
                    print('val:{} grad-{} id-{}'.format(x.data,x.grad,id(x.data)))
                if x.creator is not None:
                    add_func(x.creator) #将函数加入到函数列表中

class Function:
    def __call__(self,*inputs:Variable)->Variable:
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(self._as_array(y)) for y in ys]
        self.generation=max([x.generation for x in inputs])
        for ot in outputs:
            ot.set_creator(self)
        self.inputs=inputs #保存输入的变量
        self.outputs=outputs #保存输出的变量
        return outputs if len(outputs)>1 else outputs[0]
    
    def _as_array(self,x): #将输入转换为numpy数组
        if np.isscalar(x): #判断是否为标量
            return np.array(x)
        return x
    
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self,gy): #gy是输出方向的导数
        raise NotImplementedError()

class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y
    
    def backward(self, gy):
        return gy,gy

class Square(Function):
    def forward(self,x):
        return x**2

    def backward(self,gy): 
        x=self.inputs[0].data #已经设置为元组形式
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

def add(x0,x1):
    return Add()(x0,x1)

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



x=Variable(np.array(3))
y=add(x,x)

z=add(x,x)

u=add(y,z)

#u=add(add(x,x),add(x,x))

u.backward()


print('u.grad:{}-({})'.format(u.grad,id(u.grad)))
#print('y.grad:{}-({})'.format(y.grad,id(y.grad)))
print('x.grad:{}-({})'.format(x.grad,id(x.grad)))

print('u.val:{}-({})'.format(u.data,id(u.data)))