import random

from sqlalchemy import (create_engine, Integer, String, Column, ForeignKey,
                        UniqueConstraint)
from sqlalchemy.orm import (declarative_base, relationship,
                            sessionmaker)

Base = declarative_base()

NODE_TYPES = ("top", "middle", "bottom")
N_NODES = 3
N_EDGES = 10

class Thing(Base):
    __tablename__ = "thing"
    id = Column(Integer, primary_key=True)
    type = Column(String(10))

    @property
    def things_up(self):
        for e in sorted(self.links_up, key=lambda x: x.position):
            yield e.thing_up

    @property
    def things_down(self):
        for e in sorted(self.links_down, key=lambda x: x.position):
            yield e.thing_down

class Link(Base):
    __tablename__ = "link"
    id_up = Column(ForeignKey("thing.id"), primary_key=True)
    id_down = Column(ForeignKey("thing.id"), primary_key=True)
    position = Column(Integer, primary_key=True)

    thing_up = relationship(Thing, foreign_keys=[id_up],
        backref="links_down", order_by="Link.position")
    thing_down = relationship(Thing, foreign_keys=[id_down],
        backref="links_up", order_by="Link.position")

random.seed(0)

engine = create_engine("sqlite:///:memory:", echo=False)

Base.metadata.create_all(engine)

db = sessionmaker(engine)()

things = dict()
for t in NODE_TYPES:
    for i in range(N_NODES):
        thing = Thing(type=t)
        db.add(thing)
db.commit()

for i in range(len(NODE_TYPES)-1):
    things_lower = db.query(Thing).filter_by(type=NODE_TYPES[i+1]).all()
    things_upper = db.query(Thing).filter_by(type=NODE_TYPES[i]).all()

    for i in reversed(range(N_EDGES)):
        link = Link(thing_up=random.choice(things_upper),
                    thing_down=random.choice(things_lower),
                    position=i)

db.commit()

rs = db.query(Thing)
for thing in rs:
    print("NODE ", thing.id, thing.type)
    for l in thing.links_down:
        print(f"  v {l.position} {l.thing_down.id}")
    for l in thing.links_up:
        print(f"  ^ {l.position}, {l.thing_up.id}")

