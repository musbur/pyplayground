from typing import List
from sqlalchemy import (create_engine, Integer, CHAR, String,
                        ForeignKey)
from sqlalchemy.orm import (Mapped, mapped_column, sessionmaker,
                            DeclarativeBase, relationship)

class Base(DeclarativeBase):
    pass

class Flow(Base):
    __tablename__ = 'flow'
    id : Mapped[int] = mapped_column(primary_key=True)
    sequences: Mapped[List["Sequence"]] = relationship("Sequence")
    alarms: Mapped[List["Alarm"]] = relationship(
        primaryjoin="and_(Flow.id == foreign(Alarm.fqs_id),"
                    "Alarm.fqs == 'F')",
        overlaps="alarms")

class Sequence(Base):
    __tablename__ ='sequence'
    id : Mapped[int] = mapped_column(primary_key=True)
    flow_id : Mapped[int] = mapped_column(ForeignKey("flow.id"))
    flow : Mapped["Flow"] = relationship(back_populates="sequences")
    steps : Mapped[List["Step"]] = relationship("Step")
    alarms : Mapped[List["Alarm"]] = relationship(
        primaryjoin="and_(Sequence.id == foreign(Alarm.fqs_id),"
                    "Alarm.fqs == 'Q')",
        overlaps="alarms")

class Step(Base):
    __tablename__ = 'step'
    id : Mapped[int] = mapped_column(primary_key=True)
    sequence_id : Mapped[int] = mapped_column(ForeignKey("sequence.id"))
    sequence : Mapped["Sequence"] = relationship(back_populates="steps")
    alarms : Mapped[List["Alarm"]] = relationship(
        primaryjoin="and_(Step.id == foreign(Alarm.fqs_id),"
                    "Alarm.fqs == 'S')",
        overlaps="alarms")


class Alarm(Base):
    __tablename__ = 'alarm'
    fqs_id : Mapped[int] = mapped_column(primary_key=True)
    fqs : Mapped[str] = mapped_column(CHAR(1), primary_key=True)
    text : Mapped[str] = mapped_column(String(100))

if __name__ == '__main__':
    URI =r"mysql://dh:sunm22n@localhost/test?charset=utf8mb4"
    engine = create_engine(URI, echo=False, pool_pre_ping=True)
    Base.metadata.create_all(engine)

    db = sessionmaker(engine)()
    f = Flow()
    db.add(f)
    q = Sequence()
    a = Alarm(text="Sequence Alarm 0", fqs='Q')
    q.alarms.append(a)
    s = Step()
    f.sequences.append(q)
    q.steps.append(s)
    db.commit()

    q1 = Sequence()
    a = Alarm(text="Sequence Alarm 1", fqs='Q')
    q1.alarms.append(a)
    q1.id = q.id
    db.merge(q1)



    db.commit()

