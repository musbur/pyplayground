from collections import namedtuple
from pprint import pprint
from types import new_class

def Simple(name, slots):
    class Obj():
        __slots__ = slots

        def __init__(self, *a, **kw):
            if a:
                if len(a) == len(self.__slots__):
                    for i, k in enumerate(self.__slots__):
                        setattr(self, k, a[i])
                else:
                    raise ValueError
            for k, v in kw.items():
                setattr(self, k, v)

        def __str__(self):
            print(dir(self))
            pieces = [s + '=' + str(getattr(self, s))\
                for s in self.__slots__]
            return '%s(%s)' % (type(self).__name__, ', '.join(pieces))


    return Obj

def Slotted(name, slots):
    '''Returns a new class with named fields like namedtuple. Fields
       can be assigned new values after instance creation., but no
       new fields can be added.'''

    def _body(ns):


        # put functions into namespace
        pprint(locals())
        for k, v in locals().items():
            if type(v).__name__ == 'function':
                ns[k] = v

        ns['__slots__'] = slots
        return ns

    return new_class(name, exec_body=_body)

def Slotted1(name, slots):
    '''Returns a new class with named fields like namedtuple. Fields
       can be assigned new values after instance creation., but no
       new fields can be added.'''

    def _body(ns):

        def __init__(self, *a, **kw):
            '''Initialize with one positional arg per slot or with keywords.
               Positional args may be empty'''

            if a:
                if len(a) == len(self.__slots__):
                    for i, k in enumerate(self.__slots__):
                        setattr(self, k, a[i])
                else:
                    raise ValueError('One positional arg per slot requred')
            for k, v in kw.items():
                setattr(self, k, v)

        def __str__(self):
            pieces = [s + '=' + str(getattr(self, s, None))\
                for s in self.__slots__]
            return '%s(%s)' % (type(self).__name__, ', '.join(pieces))

        def __iter__(self): # like tuple / list
            for s in self.__slots__:
                yield getattr(self, s, None)

        def __items__(self): # like dict
            for s in self.__slots__:
                yield s, getattr(self, s, None)

        def _fields(self): # like namedtuple
            return self.__slots__

        def __getitem__(self, index): # like tuple / list
            return getattr(self, self.__slots__[index], None)

        def __hash__(self):
            return hash(tuple(self))

        # put functions into namespace
        pprint(locals())
        for k, v in locals().items():
            if type(v).__name__ == 'function':
                ns[k] = v

        ns['__slots__'] = slots
        return ns

    return new_class(name, exec_body=_body)



o1 = Slotted('X', ('a',  'b', 'c'))(a=1, b=2, c=3)


def f1(a1):
    x1 = a1
    def f2(a2):
        x2 = a2
        def f3(a3):
            x3 = a3
            print('3', locals())
        f3(x2)
        print('2', locals())
    f2(x1)
    print('1', locals())
f1(1)

print(o1)
print(o1[1])
print(hash(o1))



