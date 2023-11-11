import numpy as np
import weakref  
import weakref  


import dezero

class Variable:
    def __init__(self, data ,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name=name
    
    def cleargrad(self):
        self.grad=None

    def set_creator(self,func):
        self.creator=func
        self.generation=func.generation+1
    

    def backward(self,retain_grad=False,create_graph=False): #两个参数分别表示是否保留梯度和是否创建计算图
        if self.grad is None:
            self.grad=Variable(np.ones_like(self.data))

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

            # 反向传播的计算（主处理）
            gys=[output().grad for output in f.outputs] #使用弱引用，避免循环引用
            
            with using_config('enable_backprop',create_graph):

                gxs=f.backward(*gys) #计算出反向传播值
                if not isinstance(gxs,tuple):
                    gxs=(gxs,)
                for x,gx in zip(f.inputs,gxs):
                    if x.grad is None:
                        x.grad=gx
                    else:
                        x.grad=x.grad+gx #累加梯度,且不可写成+=，+=是in-place，会覆盖id
                    if x.creator is not None:
                        add_func(x.creator) #将函数加入到函数列表中
        
                if not retain_grad:
                    for y in f.outputs:
                        y().grad=None #使用弱引用，避免循环引用

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype #返回数据类型
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n','\n'+' '* 9)
        return 'variable('+p+')'

    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):
            shape=shape[0]
        return dezero.functions.reshape(self,shape)
    
    def transpose(self):
        return dezero.functions.transpose(self)
    
    @property
    def T(self):
        return dezero.functions.transpose(self)
    
    def sum(self,axis=None,keepdims=False):
        return dezero.functions.sum(self,axis,keepdims)




def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)

def as_array(x): #将输入转换为numpy数组
    if np.isscalar(x): #判断是否为标量
        return np.array(x)
    return x

class Config:
    enable_backprop=True

import contextlib

@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)


def no_grad():
    return using_config('enable_backprop',False)


class Function:
    def __call__(self,*inputs:Variable)->Variable:
        inputs=[as_variable(x) for x in inputs] #将输入转换为Variable

        #1.正向传播
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation=max([x.generation for x in inputs]) #设置辈分

            #2.创建连接
            for output in outputs:
                output.set_creator(self) #设置连接
            self.inputs=inputs #保存输入的变量
            self.outputs=[weakref.ref(output) for output in outputs] #保存输出的变量(使用弱引用，避免循环引用)

        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self,gy): #gy是输出方向的导数
        raise NotImplementedError()

class Add(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0+x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0,gx1

def add(x0,x1):
    x1=as_array(x1)
    return Add()(x0,x1)

class Mul(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0*x1
        return y

    def backward(self,gy):
        x0,x1=self.inputs
        gx0=gy*x1
        gx1=gy*x0
        if self.x0_shape!=self.x1_shape:
            gx0=dezero.functions.sum_to(gx0,self.x0_shape)
            gx1=dezero.functions.sum_to(gx1,self.x1_shape)
        return gx0,gx1

def mul(x0,x1):
    x1=as_array(x1)
    return Mul()(x0,x1)


class Neg(Function):
    def forward(self,x):
        return -x
    
    def backward(self,gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0-x1
        return y
    
    def backward(self,gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0,gx1

def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)

def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)


class Div(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0/x1
        return y
    
    def backward(self,gy):
        x0,x1=self.inputs
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        if self.x0_shape!=self.x1_shape:
            gx0=dezero.functions.sum_to(gx0,self.x0_shape)
            gx1=dezero.functions.sum_to(gx1,self.x1_shape)
        return gx0,gx1

def div(x0,x1):
    x1=as_array(x1)
    return Div()(x0,x1)

def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)


class Pow(Function):
    def __init__(self,c):
        self.c=c
    
    def forward(self,x):
        y=x**self.c
        return y
    
    def backward(self,gy):
        x,=self.inputs
        c=self.c
        gx=c*x**(c-1)*gy
        return gx
def pow(x,c):
    return Pow(c)(x)

class Parameter(Variable):
    pass

def setup_variable():
    Variable.__add__=add
    Variable.__radd__=add
    Variable.__mul__=mul
    Variable.__rmul__=mul
    Variable.__neg__=neg
    Variable.__sub__=sub
    Variable.__rsub__=rsub
    Variable.__truediv__=div
    Variable.__rtruediv__=rdiv
    Variable.__pow__=pow
    Variable.__getitem__ = dezero.functions.get_item


def numerical_diff(f:Function,x:Variable,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)
