from sqlalchemy.types import TypeDecorator

from sqlalchemy import create_engine, Column, Integer, Float

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MyInt(TypeDecorator):
    impl = Integer
    def process_bind_param(self, value, dialect):
        print('BIND')
        return value

    def result_processor(self, value, dialect):
        raise NotImplementedError

class T(Base):
    __tablename__ = 't'
    v1 = Column(MyInt, primary_key=True)

e = create_engine('sqlite://')
Base.metadata.create_all(e)
db = sessionmaker(e)()

t1 = T(v1=1)
db.add(t1)
db.commit()

for t in db.query(T):
    print(t.v1)

