
from dezero.core import Parameter
import numpy as np
import dezero.functions as F


import weakref

class Layer:
    def __init__(self):
        self._params=set()
    
    def __setattr__(self,name,value): #调用model.l1=Linear(10)时，会调用__setattr__方法
        if isinstance(value,(Parameter,Layer)): #添加Layer
            self._params.add(name)
        
        #__setattr__方法是在设置属性时被调用的特殊方法。实例变量的名字会作为name参数传入，而值会作为value参数传入。
        super().__setattr__(name,value)

    def __call__(self,*inputs):
        outputs=self.forward(*inputs)  #forward方法需要子类实现
        if not isinstance(outputs,tuple):
            outputs=(outputs,)
        
        self.inputs=[weakref.ref(x) for x in inputs]
        self.outputs=[weakref.ref(y) for y in outputs]
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self,inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj,Layer): #如果是Layer实例，则递归调用params方法
                yield from obj.params()
            else:
                yield obj
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
    

class Linear(Layer):
    def __init__(self,out_size,nobias=False,dtype=np.float32,in_size=None):
        super().__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.dtype=dtype

        self.W=Parameter(None,name='W')
        if self.in_size is not None: #如果指定了输入大小，则初始化参数
            self._init_W()
        
        if nobias:
            self.b=None
        else:
            self.b=Parameter(np.zeros(out_size,dtype=dtype),name='b')
    
    def _init_W(self):
        I,O=self.in_size,self.out_size
        W_data=np.random.randn(I,O).astype(self.dtype)*np.sqrt(1/I)
        self.W.data=W_data

    def forward(self,x):
        #在传播时初始化权重
        if self.W.data is None:
            self.in_size=x.shape[1]
            self._init_W()
        
        y=F.linear(x,self.W,self.b)
        return y
        
