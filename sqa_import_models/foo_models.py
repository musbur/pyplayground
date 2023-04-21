from sqlalchemy import Column, Integer

from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FooTable(Base):
    __tablename__ = 'foo'
    id = Column(Integer, primary_key=True)

class BarTable(Base):
    __tablename__ = 'bar'
    id = Column(Integer, primary_key=True)
    foo_id = ForeignKey('foo.id')
    foo = relationship(FooTable)

