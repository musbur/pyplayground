from sqlalchemy import (create_engine, Integer, String, Column, ForeignKey,
                        UniqueConstraint, inspect)
from sqlalchemy.orm import (declarative_base, relationship,
                            sessionmaker)

Base = declarative_base()

class Box(Base):
    __tablename__ = 'box'
    id = Column(Integer, primary_key=True)
    things = relationship("Thing", back_populates="box")

    def get_thing(self, color):
        db = inspect(self).session
        return (db.query(Thing)
                .filter_by(box_id=self.id, color=color)
                .one_or_none)

class Thing(Base):
    __tablename__ = 'thing'
    id = Column(Integer, primary_key=True)
    box_id = Column(ForeignKey("box.id"))
    color = Column(String(20))
    box = relationship("Box", back_populates="things")

    __table_args__ = (UniqueConstraint(box_id, color),)

engine = create_engine('sqlite:///:memory:')
# engine = create_engine('sqlite:///test1.db')
Base.metadata.create_all(engine)

db = sessionmaker(engine)()

box = Box()
thing = Thing(color="blue")
box.things.append(thing)
# db.add(box)
