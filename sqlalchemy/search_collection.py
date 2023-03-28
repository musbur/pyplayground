# Untested code -- just for illustration

class Box(Base):
    __tablename__ = 'box'
    id = Column(Integer, primary_key=True)
    things = relationship('Thing', back_populates='box')

class Thing(Base):
    __tablename__ = 'thing'
    id = Column(Integer, primary_key=True)
    box_id = ForeignKey('box.id')
    color = Column(String)
    box = relationship('Box', back_populates='things')
    __table_args__ = (UniqueConstraint(box_id, color),)

db = Session()

box = db.query(Box).filter(...).one()

# Easy to find the green thing in that box:
green_thing = (db.query(Thing)
               .filter_by(box_id=box.id,
                          color='green')
               .one_or_none())

# Question: Is there a way to find the green thing without having to
# explicitly use the session?
