from sqlalchemy import create_engine, Column, Integer, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///chain?check_same_thread=False", echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)


# initial implementation - Zackiss on 3.19
class Chain(Base):
    __tablename__ = 'Chain'
    id = Column(Integer, primary_key=True)
    attributes = Column(PickleType)


def save_block_to_chain(block: dict):
    session = Session()
    block = Chain(attributes=block)
    Chain.__table__.create(engine, checkfirst=True)
    session.add(block)
    session.commit()


def get_all_blocks_from_chain() -> list:
    session = Session()
    block = [d.content for d in session.query(Chain).all()]
    return block
