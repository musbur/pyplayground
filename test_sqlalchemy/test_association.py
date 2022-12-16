from sqlalchemy import Table, Column, Integer, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

"""
assoc = Table('assoc',
    Base.metadata,
    Column('foo_id', ForeignKey('foo.id',
                                   onupdate='CASCADE',
                                   ondelete='CASCADE'),
           primary_key=True),
    Column('bar_id', ForeignKey('bar.id',
                                      onupdate='CASCADE',
                                      ondelete='CASCADE'),
           primary_key=True),
    )
"""

class Assoc(Base):
    __tablename__ = 'assoc'
    foo_id = Column(ForeignKey('foo.id',
                               onupdate='CASCADE',
                               ondelete='CASCADE'),
                    primary_key=True)
    bar_id = Column(ForeignKey('bar.id',
                               onupdate='CASCADE',
                               ondelete='CASCADE'),
                    primary_key=True)

class Foo(Base):
    __tablename__ = "foo"
    id = Column(Integer, primary_key=True)
    data = Column(Integer)
    bars = relationship('Bar', secondary="assoc")

class Bar(Base):
    __tablename__ = "bar"
    id = Column(Integer, primary_key=True)
    data = Column(Integer)
    foos = relationship('Foo', secondary="assoc")

engine = create_engine('sqlite:///test.db')
Base.metadata.create_all(engine)
Session = sessionmaker(engine)

db = Session()
foo = Foo()
for i in range(1, 5):
    bar = Bar(data = i)
    foo.bars.append(bar)
db.add(foo)
db.commit()

rs = db.query(Foo).join(Assoc).join(Bar).all()
for r in rs:
    print(r)
