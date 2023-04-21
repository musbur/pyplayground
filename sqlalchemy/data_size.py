from datetime import datetime
from random import randint
from sqlalchemy import Integer, CHAR, DateTime, String, text
from sqlalchemy import create_engine
from sqlalchemy.orm import (Mapped, mapped_column, sessionmaker,
                            DeclarativeBase)

class Base(DeclarativeBase):
    pass

def random_dt():
    pairs = (
        (1900, 2024), (1, 12), (1, 28),
        (0, 23), (0, 59), (0, 59))
    return datetime(*(randint(*p) for p in pairs))

class Test0(Base):
    __tablename__ = 'test0'
    id: Mapped[int] = mapped_column(primary_key=True)

class Test1(Base):
    __tablename__ = 'test1'
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int]

class Test2(Base):
    __tablename__ = 'test2'
    id: Mapped[int] = mapped_column(primary_key=True)
    dt: Mapped[datetime]

class Test3(Base):
    __tablename__ = 'test3'
    id: Mapped[int] = mapped_column(primary_key=True)
    s: Mapped[str] = mapped_column(String(1000))

class Test4(Base):
    __tablename__ = 'test4'
    id: Mapped[int] = mapped_column(primary_key=True)
    s: Mapped[str] = mapped_column(CHAR(100))

class Test5(Base):
    __tablename__ = 'test5'
    id: Mapped[int] = mapped_column(primary_key=True)
    s: Mapped[str] = mapped_column(CHAR(19))

URI =r"mysql://dh:sunm22n@localhost/test"
engine = create_engine(URI)

script = """
    DROP SCHEMA IF EXISTS `test`;
    CREATE SCHEMA `test` DEFAULT CHARACTER SET ascii ;
    USE `test`;
    """
#with engine.connect() as conn:
#    conn.execute(text(script))

Base.metadata.create_all(engine)
db = sessionmaker(engine)()

for i in range(10000):
    dt = random_dt()
    things = (
        Test0(),
        Test1(ts=dt.timestamp),
        Test2(dt=dt),
        Test3(s=URI),
        Test4(s=URI),
        Test5(s=dt.strftime('%Y-%m-%s %H:%M:%S')),
        )
    for t in things:
        db.add(t)
    db.flush()
db.commit()
