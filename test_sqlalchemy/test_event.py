    import pickle
    from sqlalchemy import event, LargeBinary, Column, create_engine, Integer
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm.attributes import flag_modified

    Base = declarative_base()

    class Thing(Base):
        __tablename__ = 'thing'
        id = Column(Integer, primary_key=True)
        _data = Column(LargeBinary(1000))

    @event.listens_for(Thing, 'load')
    def load_thing(target, context):
        target.data = pickle.loads(target._data)
        flag_modified(target, '_data') # doesn't work
        print('LOAD', target.data)

    @event.listens_for(Thing, 'before_insert')
    def insert_thing(mapper, connection, target):
        target._data = pickle.dumps(target.data)
        print('INSERT', target.data)

    @event.listens_for(Thing, 'before_update')
    def update_thing(mapper, connection, target):
        target._data = pickle.dumps(target.data)
        print('UPDATE', target.data)

    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine)

    # Make a Thing and put it into DB
    db = Session()
    t = Thing(id=1)
    t.data = {'a': 1}
    db.add(t)
    db.commit() # insert_thing() gets called
    db.expunge_all()

    t = db.query(Thing).get((1)) # LOAD {'a': 1}
    t.data = {'a': 2}
    db.commit() # update_thing() does NOT get called
    db.expunge_all()

    t = db.query(Thing).get((1)) # LOAD {'a': 1}
    t.data = {'a': 3}
    flag_modified(t, '_data') # works!
    db.commit() # update_thing() gets called
    db.expunge_all()

    t = db.query(Thing).get((1)) # LOAD {'a': 3}
