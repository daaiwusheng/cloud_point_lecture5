import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable


def test_torch():
    test_max_pool = nn.MaxPool1d(5, 5)
    input = torch.randn(3, 9, 5)
    output = test_max_pool(input)
    print(input)
    print(output)
    print('-' * 30)
    t = torch.max(input, 1, keepdim=True)[0]
    print(t)
    print(t.view(-1, 5))


def test_np():
    t_matrix = np.eye(3, dtype=np.float32)
    print(t_matrix)


def test_torch_variable():
    inputs = Variable(torch.randn(2, 2))
    print(inputs.is_cuda)  # will return false
    inputs = Variable(torch.randn(2, 2).cuda())
    print(inputs.is_cuda)  # returns true


def test_torch_argmax():
    a = torch.randn(4, 4)
    print(a)
    b = torch.argmax(a)
    print(b)


def test_torch_random_seed():
    torch.manual_seed(10)
    print(torch.rand(3))


def test_torch_backward_opt_step():
    x = torch.tensor([1., 2.], requires_grad=True)
    optim = torch.optim.SGD([x], lr=0.001)  # 随机梯度下降， 学习率0.001
    for i in range(3):
        y = 100 * x
        print(f'第 {i+1} 次迭代')
        loss = y.sum()
        # Compute gradients of the parameters respect to the loss
        print('反向传播前梯度: ', x.grad)
        # optim.zero_grad()
        loss.backward()
        print('反向传播后梯度: ', x.grad)
        print('更新前的x: ', x)
        optim.step()  # 更新x
        print('更新后的x: ', x)


if __name__ == '__main__':
    test_torch_backward_opt_step()
