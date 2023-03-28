from sqlalchemy import (create_engine, Integer, String, Column, ForeignKey,
                        UniqueConstraint)
from sqlalchemy.orm import (declarative_base, relationship,
                            sessionmaker)

Base = declarative_base()

class Container(Base):
    __tablename__ = 'box'
    id = Column(Integer, primary_key=True)

    def merge_boxes(self, db):
        for box in self.boxs:
            old_box = (db.query(Thing)
                         .filter_by(box_id=self.id,
                                    color=box.color)
                         .one_or_none())
            if old_box:
                box.id = old_box.id
                db.merge(box)

class Box(Base):
    __tablename__ = 'box'
    id = Column(Integer, primary_key=True)
    cintainer_id = Column(ForeignKey("container.id"))
    things = relationship("Thing", back_populates="box")

    def merge_things(self, db):
        for thing in self.things:
            old_thing = (db.query(Thing)
                         .filter_by(box_id=self.id,
                                    color=thing.color)
                         .one_or_none())
            if old_thing:
                thing.id = old_thing.id
                db.merge(thing)

    __table_args__ = (UniqueConstraint(container_id, label),)

class Thing(Base):
    __tablename__ = 'thing'
    id = Column(Integer, primary_key=True)
    box_id = Column(ForeignKey("box.id"))
    color = Column(String(20))
    box = relationship("Box", back_populates="things")
    extras = relationship("Extra", back_populates="thing")

    __table_args__ = (UniqueConstraint(box_id, color),)

class Extra(Base):
    __tablename__ = 'extra'
    thing_id = Column(ForeignKey("thing.id"), primary_key=True)
    thing = relationship("Thing", back_populates="extras")

def get_box():
    box = Box()
    box.label = 100
    for color in ("red", "green", "yellow"):
        thing = Thing(color=color)
        box.things.append(thing)
    return box

engine = create_engine('sqlite:///:memory:')
# engine = create_engine('sqlite:///test1.db')
Base.metadata.create_all(engine)

db = sessionmaker(engine)()

# set up some data
box = get_box()
db.add(box)
for t in box.things:
    t.extras.append(Extra())
db.commit()

# Here comes the second call.
box = get_box()
box.things.append(Thing(color="blue"))

old_box = db.query(Box).filter_by(label=100).one()
box.id = old_box.id
box.merge_things(db)
db.merge(box)
db.commit()

