class foo():
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def __hash__(self):
        return hash((self.b, self.c))

    def __eq__(self, other):
        return self.b == other.b and self.c == other.c

    def __repr__(self):
        return str((self.a, self.b, self.c))

a = foo(1, 2, 3)
b = foo(2, 2, 3)
c = foo(3, 2, 3)


