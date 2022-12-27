from sqlalchemy import create_engine, CHAR, Column, Integer,\
    String, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Thing(Base):
    __tablename__ = 'thing'
    id            = Column(Integer, primary_key=True)
    name          = Column(String(255))
    up = association_proxy('coll_up', 'up')
    down = association_proxy('coll_down', 'down')

class Link(Base):
    __tablename__ = 'link'
    id_up         = Column(Integer, ForeignKey('thing.id'),
                           primary_key=True)
    id_down       = Column(Integer, ForeignKey('thing.id'),
                          primary_key=True)
    position      = Column(Integer)
    up = relationship('Thing',
                      primaryjoin=id_up == Thing.id,
                      order_by='Link.position',
                      backref='coll_down')
    down = relationship('Thing',
                        primaryjoin=id_down == Thing.id,
                        order_by='Link.position',
                        backref='coll_up')

if __name__ == '__main__':
# I know that for convenient append()ing stuff I'd need an __init__()
# method, but see comments below

    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine)

    if __name__ == '__main__':
        db = Session()
        root = Thing(name='Root thing')
        db.add(root)
# inserting "reversed" to have something to sort
        for i in reversed(range(5)):
            t = Thing(name='Thing %d' % i)
# If Link had an __init__ method I could use root.append(t), but...
            db.add(t)
# ...in order to set position, I need to explicitly use the association
# object anyway
            l = Link(up=root, down=t, position=i)
            db.add(l)
        db.commit()

# OK, so far the association_proxy construct hasn't been of any use. Let's
# see about retrieving the data...

        root = db.query(Thing).filter_by(name='Root thing').one()
        for thing in root.down:
            print(thing.name)
# This, as expected. prints the elements in the order in which they were
# inserted (which is "reversed"). How can I sort them by the "position"
# property of the "Link" element, or is the associaton object useless in
# this scenario?
