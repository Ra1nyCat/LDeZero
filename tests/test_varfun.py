#python -m unittest discover tests  运行tests文件夹下所有测试文件

import unittest
from VarFun import Variable,square
import numpy as np


def numerical_diff(f,x:Variable,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)

class SquareTest(unittest.TestCase):

    def test_forward(self):
        x=Variable(np.array(2.0))
        y=square(x)
        expected=np.array(4.0)
        self.assertEqual(y.data,expected)

    def test_backward(self):
        x=Variable(np.array(3.0))
        y=square(x)
        expected=np.array(6.0)
        y.backward()
        self.assertEqual(x.grad,expected)

    
    def test_gradient_check(self):

        for i in range(20):
            x=Variable(np.random.rand(1))
            y=square(x)
            y.backward()
            num_grad=numerical_diff(square,x)
            flg=np.allclose(x.grad,num_grad,rtol=1e-5,atol=1e-8)
            self.assertTrue(flg)

# unittest.main() #运行时可以省略掉-m unittest