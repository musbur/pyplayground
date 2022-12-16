class Foo():
    x : set = set()

    def __init__(self, s):
        if s:
            self.x.add(s)


foo = Foo(None)
print(foo.x) # prints 'set()'
bar = Foo('abc')
print(foo.x) # prints '{'abc'}
