from functools import wraps

def deco(func):
    def wrapper(*a, **k):
        return func(x=1, y=2, *a, **k)
    return wrapper


@deco
def f(y, x, b):
    pass

f(b=7)

def arg_deco(arg):
    def deco(func):
        @wraps(func)
        def wrapper(*a, **k):
            return func(*a, **k)
        return wrapper
    return deco
