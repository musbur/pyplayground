from sqlalchemy import create_engine, Integer, Column
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()

class Foo(Base):
    __tablename__ = 'foo'
    id = Column(Integer, primary_key=True)

    @hybrid_property
    def id_check(self):
        return self.id & 0x01 == 0x01

    @id_check.expression
    def id_check(cls):
        return cls.id.op('&')(0x01) == 0x01

engine = create_engine('sqlite://')
db = sessionmaker(engine)()

q = db.query(Foo).filter(Foo.id_check)

print(q)
