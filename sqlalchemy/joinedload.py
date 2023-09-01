from sqlalchemy import Column, Integer, String, create_engine, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, with_loader_criteria
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Child(Base):
    __tablename__ = 'child'
    id = Column(Integer, primary_key=True)
    parent_id = Column(ForeignKey('parent.id'))
    name = Column(String(20))
    parent = relationship('Parent', back_populates='children')

class Parent(Base):
    __tablename__ = 'parent'
    id = Column(Integer, primary_key=True)
    children = relationship('Child', back_populates='parent')

# set up DB
engine = create_engine('sqlite://')
Base.metadata.create_all(engine)
db = sessionmaker(engine)()

# populate DB
parent = Parent()
db.add(parent)
parent.children.extend([Child(name="Eric"), Child(name="Emmy")])

parent = Parent()
db.add(parent)
parent.children.extend([Child(name="Tom"), Child(name="Bob")])
db.commit()

q = (db.query(Parent)
     .options(joinedload

print_statement(q)

for parent in q:
    print(parent.id)
    for child in parent.children:
        print('  ', child.name)


