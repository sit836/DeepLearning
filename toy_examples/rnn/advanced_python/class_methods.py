"""
    https://zhuanlan.zhihu.com/p/28010894
"""


class A:
    def m1(self, n):
        print("self:", self)

    @classmethod
    def m2(cls, n):
        print("cls:", cls)

    @staticmethod
    def m3(n):
        print(n)


a = A()
a.m1(1)  # self: <__main__.A object at 0x000001E596E41A90>
A.m2(1)  # cls: <class '__main__.A'>
A.m3(1)
a.m3(10)

print(A.m1)  # In py2, A.m1 outputs <unbound method A.m1>
print(a.m1)

# A.m1(a, 1) is equivalent to a.m1(1)
A.m1(a, 1)
a.m1(1)

# A.m1(1)  # TypeError: m1() missing 1 required positional argument: 'n'

print(A.m2)  # <bound method A.m2 of <class '__main__.A'>>

# Fact1. python可以通过实例对象a找到它所属的类是A，找到A之后自动绑定到 cls
print(a.m2)  # <bound method A.m2 of <class '__main__.A'>>
# A.m2 is equivalent to a.m2

# By Fact 1, 我们可以在实例方法中通过使用 self.m2()这种方式来调用类方法和静态方法。
# def m1(self, n):
#     print("self:", self)
#     self.m2(n)

