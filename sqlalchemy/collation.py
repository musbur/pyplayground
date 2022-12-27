import random
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, CHAR, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class XTab1(Base):
    __tablename__ = 'xtab1'
    id            = Column(Integer, primary_key=True)
    col           = Column(String(100, collation='ascii_bin'))

class XTab2(Base):
    __tablename__ = 'xtab2'
    id            = Column(Integer, primary_key=True)
    col           = Column(String(400))

class XTab3(Base):
    __tablename__ = 'xtab3'
    id            = Column(Integer, primary_key=True)
    col           = Column(CHAR(255, collation='ascii_bin'))

class XTab4(Base):
    __tablename__ = 'xtab4'
    id            = Column(Integer, primary_key=True)
    col           = Column(CHAR(255))

engine = create_engine('mysql://dham@deham01in015/dh_test')
Base.metadata.create_all(engine)

Session = sessionmaker(engine)
db = Session()

pop = 'ABCDEFGHIJKLMNOPQRäüöVWXYZabcdefghijklmnopqrtsuvwxyz0123456789'
for i in range(10000):
    s = ''.join(random.choices(pop, k=100))
    for u in '1234':
        cls = locals()['XTab' + u]
        x = cls(col=s)
        db.add(x)
db.commit()
