"""
    Reference:
    https://wiki.python.org/moin/Iterator
    https://wiki.python.org/moin/Generators
"""

"""
Definitions

Iterator: any object whose class has a __next__ method and an __iter__ method that does return self.

Generator: Every generator is an iterator, but not vice versa. A generator is built by calling a function that
has one or more yield expressions, and is an object that meets the definition of an iterator.
"""
listA = ['a', 'e', 'i']

my_iter = listA.__iter__()
print(my_iter.__next__())
print(my_iter.__next__())
print(my_iter.__next__())


###
class Counter:
    def __init__(self, start, end):
        self.num = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.num - 1


c1 = Counter(start=2, end=5)
for i in c1:
    print("Eating more Pizzas, counting ", i, end="\n")


###
def first_n_naive(n: int) -> list:
    num, nums = 0, []
    while num <= n:
        nums.append(num)
        num += 1
    return nums


class FirstN:
    def __init__(self, n):
        self.n = n
        self.num = 0

    def __iter__(self):
        return self  # see class_methods.py

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
        yield num
        num += 1


print(f'sum of first_n_naive: {sum(first_n_naive(100))}')

print(f'sum of FirstN: {sum(FirstN(100))}')

print(f'sum of firstn_generator: {sum(firstn_generator(100))}')
