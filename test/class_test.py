
class A(object):
    def __init__(self):
        self.a = 1
        self.b = 2

    def print_a(self):
        print(self.a)

    def print_b(self):
        print(self.b)


class B(A):
    pass


s = B()
s.print_a()