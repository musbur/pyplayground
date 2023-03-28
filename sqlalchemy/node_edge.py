import random

from sqlalchemy import (create_engine, Integer, String, Column, ForeignKey,
                        UniqueConstraint)
from sqlalchemy.orm import (declarative_base, relationship,
                            sessionmaker)

Base = declarative_base()

class Node(Base):
    __tablename__ = 'node'
    id = Column(Integer, primary_key=True)
    type = Column(String(10))

    def __str__(self):
        return f"{self.id} {self.type}"

    @property
    def nodes_up(self):
        for e in self.edges_up:
            yield e.node_up

    @property
    def nodes_down(self):
        for e in self.edges_down:
            yield e.node_down

class Edge(Base):
    __tablename__ = 'edge'
    id_up = Column(ForeignKey('node.id'), primary_key=True)
    id_down = Column(ForeignKey('node.id'), primary_key=True)
    position = Column(Integer, primary_key=True)

    node_up = relationship(Node, foreign_keys=[id_up],
        backref='edges_down', order_by="Edge.position")
    node_down = relationship(Node, foreign_keys=[id_down],
        backref='edges_up', order_by="Edge.position")


random.seed(0)

engine = create_engine('sqlite:///:memory:')

Base.metadata.create_all(engine)

db = sessionmaker(engine)()

nodes = dict()
for t in ('top', 'middle', 'down'):
    for i in range(10):
        node = Node(type=t)
        db.add(node)
db.commit()

for upper, lower in (('top', 'middle'), ('middle', 'down')):
    nodes_upper = db.query(Node).filter_by(type=upper).all()
    nodes_lower = db.query(Node).filter_by(type=lower).all()

    for i in range(10):
        edge = Edge(node_up=random.choice(nodes_upper),
                    node_down=random.choice(nodes_lower),
                    position=i)

db.commit()

rs = db.query(Node)
for node in rs:
    print('NODE ', node)
    for n in node.nodes_down:
        print('  v ', n)
    for n in node.nodes_up:
        print('  ^ ', n)


