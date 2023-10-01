import torch

"""
    Dive into deep learning, 2.5
"""

"""
    2.5.1 A Simple Function
"""
x = torch.arange(4.0)
print(f'x: {x}')

# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
print(f'x.grad: {x.grad}')  # The gradient is None by default

y = 2 * torch.dot(x, x)
print(f'y: {y}')

# take the gradient of y with respect to x by calling its backward method
# y.backward(retain_graph=True)
y.backward()
print(f'x.grad: {x.grad}')

# y.backward()
# print(f'x.grad: {x.grad}')

# y.backward(retain_graph=True)
# print(f'x.grad: {x.grad}')

# Now let’s calculate another function of x and take its gradient
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
print(f'x.grad: {x.grad}')

"""
    2.5.2 Backward for Non-Scalar Variables
    https://zhuanlan.zhihu.com/p/65609544
"""
# we need to provide some vector v such that backward will compute v.⊤@dydx rather than dydx
x.grad.zero_()
y = x * x
y.sum().backward()
print(f'x.grad: {x.grad}')

"""

    2.5.3 Detaching Computation
"""
# Suppose z = x * y and y = x * x. We want to know the direct influence of x on
# z rather than the influence conveyed via y.
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
print(f'x: {x}')
print(f'y: {y}')
print(f'z: {z}')
z.sum().backward()
print(f'x.grad: {x.grad}')
# taking the gradient of z = x * u will yield the result x, (not 3 * x * x as you might have expected since z = x * x * x).

x.grad.zero_()
z = x * x * x
z.sum().backward()
print(x.grad)

"""
    2.5.4 Gradients and Python Control Flow
"""


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(f'a.grad: {a.grad}')
print(f'd/a: {d / a}')

"""
    (i) attach gradients to those variables with respect to which we desire derivatives; 
    (ii) record the computation of the target value; 
    (iii) execute the backpropagation function; 
    (iv) access the resulting gradient
"""
