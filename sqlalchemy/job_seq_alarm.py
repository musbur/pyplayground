from sqlalchemy import (Column, Integer, ForeignKey, create_engine, String,
                        UniqueConstraint)
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import event

Base = declarative_base()

class HasAlarm():
    def append_alarm(self, message):
        alarm = Alarm(message=message, idx=self.alarm_idx)
        self.alarms.append(alarm)
        self.alarm_idx += 1

class Job(Base, HasAlarm):
    __tablename__ = "job"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    sequences = relationship('Sequence')
    alarms = relationship('Alarm', back_populates='job',
                          cascade='all, delete-orphan')

    def __init__(self, **kw):
        super().__init__(**kw)
        self.alarm_idx = 0

    def link_alarms(self):
        for sq in job.sequences:
            for al in sq.alarms:
                al.job = self

    def merge_sequences(self, db):
        for sq in self.sequences:
            old_sq = (db.query(Sequence)
                      .filter_by(job_id=self.id,
                                 name=sq.name)
                      .one_or_none())
            if old_sq:
                sq.id = old_sq.id
                db.merge(sq)

class Sequence(Base, HasAlarm):
    __tablename__ = "sequence"
    id = Column(Integer, primary_key=True)
    job_id = Column(ForeignKey("job.id"))
    name = Column(String(100))
    job = relationship(Job, back_populates='sequences')
    alarms = relationship('Alarm', back_populates='sequence',
                          cascade='all, delete-orphan')

    __table_args__ = (UniqueConstraint(job_id, name),)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.alarm_idx = 0

class Alarm(Base):
    __tablename__ = 'alarm'
    job_id = Column(ForeignKey('job.id'), primary_key=True, default=0)
    seq_id = Column(ForeignKey('sequence.id'), primary_key=True, default=0)
    idx = Column(Integer, primary_key=True)
    message = Column(String(100))
    job = relationship('Job', back_populates='alarms')
    sequence = relationship('Sequence', back_populates='alarms')

engine = create_engine('sqlite:///job_seq_alarm.db')
Base.metadata.create_all(engine)
db = sessionmaker(engine)()

def make_job():
    job = Job(name='First Job')
    for i in range(3):
        sequence = Sequence(name=f'Seq {i}')
        job.sequences.append(sequence)

    job.append_alarm('Job problem')
#    job.sequences[0].append_alarm('Seq prob 1')
#    job.sequences[0].append_alarm('Seq prob 2')
    return job

job = make_job()
job.link_alarms()
db.add(job)
db.commit()


new_job = make_job()
new_job.link_alarms()
new_job.id = 1
new_job.merge_sequences(db)
db.merge(new_job)
db.commit()


