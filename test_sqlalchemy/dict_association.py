from sqlalchemy import create_engine, Column, Integer,\
    String, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship, sessionmaker, backref
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Basket(Base):
    __tablename__ = 'basket'
    id            = Column(Integer, primary_key=True)
    contents = association_proxy('basket_contents', 'id')

class Link(Base):
    __tablename__ = 'link'
    item_id       = Column(ForeignKey('item.id'), primary_key=True)
    basket_id     = Column(ForeignKey('basket.id'), primary_key=True)
    quantity      = Column(Integer)
    basket = relationship(Basket, backref=backref('basket_contents',
        collection_class=attribute_mapped_collection('quantity')))
    item = relationship('Item')
    name = association_proxy('item', 'name')

    def __init__(self, name, quantity):
        # this doesn't work b/c it calls Item.__init__() rather than
        # looking for an existing Item
        self.name = name
        self.quantity = quantity

class Item(Base):
    __tablename__ = 'item'
    id            = Column(Integer, primary_key=True)
    name          = Column(String(10), unique=True)
    weight        = Column(Integer)
    color         = String(10)

engine = create_engine('sqlite://')
Base.metadata.create_all(engine)
Session = sessionmaker(engine)

db = Session()

egg = Item(name='egg', weight=50, color='white')

b = Basket()

# fails because in Link.__init__(), SQLAlchemy wants to create a new Item
# rather than using the existing one.
b.contents['egg'] = 6

db.add(b)

db.commit()
