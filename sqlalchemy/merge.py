from sqlalchemy import (create_engine, Integer, String, Column, ForeignKey,
                        UniqueConstraint, inspect)
from sqlalchemy.orm import (declarative_base, relationship,
                            sessionmaker)

Base = declarative_base()

class Thing(Base):
    __tablename__ = 'thing'
    id = Column(Integer, primary_key=True)
    a = Column(Integer)
    b = Column(Integer)

engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)

db = sessionmaker(engine)()

t1 = Thing(id=1, a=2, b=3)
db.add(t1)
db.commit()

t2 = Thing(id=1, a=4)

print(t1.a, t1.b)

t3 = db.merge(t2)

db.commit()

print(t1.a, t1.b)
print(t2.a, t2.b)
print(t3.a, t3.b)
print(t3 is t1)

