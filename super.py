class Foo():
    classvar = None


foo = Foo()
print(foo.classvar)

bar = Foo()
bar.classvar = 'abc'
print(foo.classvar)
print(bar.classvar)
