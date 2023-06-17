from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import sessionmaker, relationship, foreign, remote
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


_alarm_join = dict(primaryjoin="Parent.id == Child.parent_id",
                   foreign_keys="Child.parent_id")

class Child(Base):
    __tablename__ = 'child'
    parent_id = Column(Integer, primary_key=True, default=0)
    other_id = Column(Integer, primary_key=True, default=0)

    parent = relationship('Parent', back_populates='children',
                          **_alarm_join)

class Parent(Base):
    __tablename__ = 'parent'
    id = Column(Integer, primary_key=True)
    children = relationship('Child', back_populates='parent',
                            **_alarm_join)

engine = create_engine('sqlite://')
Base.metadata.create_all(engine)
db = sessionmaker(engine)()

parent = Parent()
child = Child()
other = Child(other_id=1)
parent.children.append(child)
db.add(parent)
db.add(other)
db.commit()


