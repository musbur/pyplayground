from sqlalchemy import Column, Integer

from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from .foo_models import FooTable

Base = declarative_base()

class BarTable(Base):
    __tablename__ = 'bar'
    id = Column(Integer, primary_key=True)
    foo_id = ForeignKey('foo.id')
    foo = relationship(FooTable)

