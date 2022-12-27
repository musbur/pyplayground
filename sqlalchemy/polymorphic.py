from sqlalchemy import create_engine, CHAR, Column, Integer,\
    String, ForeignKey

from sqlalchemy.ext.associationproxy import association_proxy

from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Recipe(Base):
    __tablename__ = 'recipe'
    id            = Column(Integer, primary_key=True)
    type          = Column(CHAR(1), index=True,
                           doc='F:Flow, Q:Sequence, S:Step')
    name          = Column(String(255))
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': None
    }

class Link(Base):
    __tablename__ = 'link'
    id_up         = Column(Integer, ForeignKey('recipe.id'),
                           primary_key=True)
    id_down       = Column(Integer, ForeignKey('recipe.id'),
                          primary_key=True)
    position      = Column(Integer)
    up = relationship('Recipe', primaryjoin=id_up == Recipe.id)
    down = relationship('Recipe', primaryjoin=id_down == Recipe.id)

class Flow(Recipe):
    sequences = relationship('Link', back_populates='up',
        primaryjoin='Link.id_up == Flow.id')
    __mapper_args__ = {
        'polymorphic_identity': 'F',
    }

class Sequence(Recipe):
    flows = relationship('Link', back_populates='up',
        primaryjoin='Link.id_down == Sequence.id')
    steps = relationship('Link', back_populates='down',
        primaryjoin='Link.id_up == Sequence.id')
    __mapper_args__ = {
        'polymorphic_identity': 'Q',
    }

class Step(Recipe):
    sequences = relationship('Link', back_populates='down',
        primaryjoin='Link.id_down == Step.id')
    __mapper_args__ = {
        'polymorphic_identity': 'S',
    }

engine = create_engine('sqlite:///test.db')
Base.metadata.create_all(engine)
Session = sessionmaker(engine)

if __name__ == '__main__':
    db = Session()
    step = Step(name='Step1')
    seq = Sequence(name='Sequence1')
    flow = Flow(name='Flow1')
    flow.sequences.append(Link(up=flow, down=seq))
    seq.steps.append(Link(up=seq, down=step))
    db.add(flow)
    db.commit()

    for flow in db.query(Flow):
        print(flow.name)
        for l1 in flow.sequences:
            seq = l1.down
            print(' ', seq.name)
            for l2 in seq.steps:
                step = l2.down
                print('  ', step.name)
