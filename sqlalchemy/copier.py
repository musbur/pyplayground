from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, inspect


Base = declarative_base()

class A(Base):
    __tablename__ = "a"
    id = Column(Integer, primary_key=True)
    a = Column(String(10))
    b = Column(Integer)
    c = Column(Integer)
    d = Column(Integer)
    e = Column(Integer)
    f = Column(Integer)

class B(Base):
    __tablename__ = "b"
    id = Column(Integer, primary_key=True)
    a = Column(String(20))
    b = Column(Integer)
    c = Column(String(19))
    d = Column(Integer)
    t = Column(Integer)
    q = Column(Integer)

def copier(cls_from, cls_to, exclude={}, typecheck=True):
    i_from = inspect(cls_from).columns
    i_to = inspect(cls_to).columns

    colnames = ((set(i_from.keys())
                 & set(i_to.keys()))
                - set(exclude))

    if typecheck:
        for c in colnames:
            f = str(i_from[c].type)
            t = str(i_to[c].type)
            if f != t:
                raise TypeError (f"Column {c}: {f} <> {t}")

    def copy(inst_from, inst_to):
        for c in colnames:
            setattr(inst_to, c, getattr(inst_from, c))

    return copy

def print_thing(thing):
    cols = inspect(type(thing)).columns.keys()
    for c in cols:
        print(c, getattr(thing, c))


a_to_b = copier(A, B, exclude={"t"}, typecheck=False)

a = A(b=3, c=4)
b = B(b=20, d=1)

print("a")
print_thing(a)
print("b")
print_thing(b)

a_to_b(a, b)

print("\na")
print_thing(a)
print("b")
print_thing(b)
