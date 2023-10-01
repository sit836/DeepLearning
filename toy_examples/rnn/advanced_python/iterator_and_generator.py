"""
    Reference:
    https://wiki.python.org/moin/Iterator
    https://wiki.python.org/moin/Generators
"""

"""
An iterable object is an object that implements __iter__, which is expected to return an iterator object.

An iterator object implements __next__, which is expected to return the next element of the iterable object 
that returned it, and to raise a StopIteration exception when no more elements are available.

"""
def first_n_naive(n: int) -> list:
    num, nums = 0, []
    while num <= n:
        nums.append(num)
        num += 1
    return nums


class FirstN(object):
    def __init__(self, n):
        self.n = n
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num <= self.n:
            cur, self.num = self.num, self.num + 1
            return cur
        raise StopIteration()


def firstn_generator(n: int):
    num = 0
    while num <= n:
        print(f'num: {num}')
        yield num
        num += 1


print(f'sum of first_n_naive: {sum(first_n_naive(100))}')

print(f'sum of FirstN: {sum(FirstN(100))}')

print(f'sum of firstn_generator: {sum(firstn_generator(100))}')
