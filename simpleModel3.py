#!/usr/bin/python
# coding: utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain

# 論理演算のXORはパーセプトロンが2層ないと解けない
# 一方で、層を6層に増やしたりシても、逆伝播で勾配が消失してしまうことがわかっている

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1=L.Linear(2, 2),
            l2=L.Linear(2, 1),
        )
    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)

model = Model()
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

x = Variable(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32))
t = Variable(np.array([[0],[1],[1],[0]],dtype=np.float32))

for i in range(0,3000):
    optimizer.zero_grads()
    y = model(x)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()

    print("loss:",loss.data)

print(y.data)