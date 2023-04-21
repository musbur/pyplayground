from typing import List
from sqlalchemy import (create_engine, Integer, CHAR, String,
                        ForeignKey, Column)
from sqlalchemy.orm import (sessionmaker,
                            declarative_base, relationship)

Base = declarative_base()

class Thing(Base):
    __tablename__ = 'thing'
    name = Column(CHAR(3), primary_key=True)
    seq = Column(Integer, primary_key=True)
    id = Column(Integer, autoincrement=True, nullable=False, index=True)

if __name__ == '__main__':
    URI =r"mysql://dh:sunm22n@localhost/test?charset=utf8mb4"
    engine = create_engine(URI, echo=True, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    db = sessionmaker(engine)()

    for i in range(11, 20):
        t = Thing(name='abc', seq=i)
        db.add(t)

    db.commit()

