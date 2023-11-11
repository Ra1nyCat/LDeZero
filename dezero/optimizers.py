
class Optimizer:

    def __init__(self):
        self.target=None #Model或Layer实例
        self.hooks=[]

    
    def setup(self,target):
        self.target=target
        return self
    
    def update(self):
        #将None之外的参数汇总到列表中
        params=[p for p in self.target.params() if p.grad is not None]

        #预处理
        for f in self.hooks:
            f(params)
        
        #更新参数
        for param in params:
            self.update_one(param)
    

    def update_one(self,param):
        raise NotImplementedError()
    
    def add_hook(self,f):
        self.hooks.append(f)
    


class SGD(Optimizer):

    def __init__(self,lr=0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self,param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):

    def __init__(self,lr=0.01,momentum=0.9):
        super().__init__()
        self.lr=lr
        self.momentum=momentum
        self.vs=None
    
    def update_one(self,param):
        v=self.vs.get(param.data.shape,param.data.dtype)
        v=self.momentum*v-self.lr*param.grad.data
        param.data+=v
    
    def add_hook(self,f):
        self.hooks.append(f)
        self.vs=utils.Dictionary()
        for param in self.target.params():
            self.vs[param]=np.zeros_like(param.data)