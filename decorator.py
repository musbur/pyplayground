from functools import wraps

def deco(func):
    def wrapper(*a, **k):
        return func(x=1, y=2, *a, **k)
    return wrapper


@deco
def f(y, x, b):
    pass

f(b=7)

def needs_key(key):
    def inner_decorator(func):
        def wrapped(dict_arg):
            if not isinstance(dict_arg, dict):
                print('Call me with a dict')
                return None
            # where do I get the key?
            if not key in dict_arg:
                print('Key is not in dict')
                return None
            return func(dict_arg)
        return wrapped
    return inner_decorator

@needs_key('a')
def func_a(d):
    return d['a']

aa = {'a': 'A'}
bb = {'b': 'A'}

print(func_a(aa))
print(func_a(bb))

def func_b(d):
    return d['b']

fb = needs_key(func_b)('a')

fb(aa)

