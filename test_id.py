class X():

    def __init__(self, data):
        super().__setattr__('data', data)

    def __getattr__(self, key):
        return getattr(self.data, key)

    def __setattr__(self, key, value):
        setattr(self.data, key, value)

class OBJ():
    pass

x = X(OBJ())

x.abc = 'abc'


print(x.abc)
