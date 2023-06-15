class WithMessages():

    def __init__(self):
        self.messages = dict()
        self.idx = 0

    def add_message(self, msg):
        self.messages[self.idx] = msg
        self.idx += 1


class SomeClass():

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

class Foo(SomeClass, WithMessages):

    def __init__(self, **kw):
        SomeClass.__init__(self, **kw)
        WithMessages.__init__(self)


foo = Foo(name="abc")

foo.add_message("I am here")
foo.add_message("I am still here")

print(foo.name)
print(foo.messages)
