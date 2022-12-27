from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, TIMESTAMP, Column, Integer, func
Base = declarative_base()

class Foo(Base):
    __tablename__ = 'mixmax'
    id = Column(Integer, primary_key=True)
    ts = Column(TIMESTAMP, nullable=False)


engine = create_engine("mysql://dh:sunm22n@localhost/test")
Base.metadata.create_all(bind=engine)
