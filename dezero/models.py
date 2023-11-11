from dezero import Layer
from dezero import utils

class Model(Layer):
    def plot(self,*inputs,to_file='model.png'):
        y=self.forward(*inputs)
        return utils.plot_dot_graph(y,to_file=to_file)
    
    def save(self,path):
        self.to_cpu()
        params=[p.data for p in self.params()]
        np.savez_compressed(path,*params)
    
    def load(self,path):
        npz=np.load(path)
        params=[p.data for p in self.params()]
        for p,t in zip(params,npz):
            p[...]=t
    
    def parameters(self):
        for name in self._params:
            obj=self.__dict__[name]
            if isinstance(obj,Layer):
                yield from obj.parameters()
            else:
                yield obj
    
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
    
    def train(self):
        for param in self.params():
            param.requires_grad=True
    
    def eval(self):
        for param in self.params():
            param.requires_grad=False
    

import dezero.functions as F
import dezero.layers as L

class MLP(Model):
    def __init__(self,fc_out_sizes,activation=F.sigmoid):
        super().__init__()
        self.activation=activation
        self.layers=[]
        
        for i,out_size in enumerate(fc_out_sizes):
            layer=L.Linear(out_size)
            setattr(self,'l'+str(i),layer)
            self.layers.append(layer)
    
    def forward(self,x):
        for l in self.layers[:-1]:
            x=self.activation(l(x))
        return self.layers[-1](x)


