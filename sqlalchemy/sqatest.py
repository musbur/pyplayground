from sqlalchemy import Column, Integer, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Foo(Base):
    __tablename__ = "foo"
    id = Column(Integer, primary_key=True)
    data = Column(Integer)
    bar = relationship("Bar", back_populates="foo")

class Bar(Base):
    __tablename__ = "bar"
    id = Column(Integer, primary_key=True)
    foo_id = Column(ForeignKey("foo.id"))
    data = Column(Integer)
    foo = relationship("Foo", back_populates="bar")

class Bar2(Bar):
    def do_stuff2(self):
        print("Doing stuff2")

def do_stuff(instance):
    print("Doing stuff")

Bar.do_stuff = do_stuff

foo = Foo()
foo.bar = [Bar("this is me")]
foo.bar[0].do_stuff()
foo.bar[0].do_stuff2() # fails
