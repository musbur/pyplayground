from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class RecipeRegistry():
    def __init__(self, db):
        self.db = db
        self.data = dict()

    def __get__(self, key):
        value = self.data.get(key)
        if value:
            return value
        

class Recipe(Base):
    __tablename = "recipe"
    id          = Column(Integer, primary_key=True)
    name        = Column(String(200), unique=True)

class Sequence(Base):
    __tablename = "sequence"
    id          = Column(Integer, primary_key=True)
    recipe_id   = Column(ForeignKey("recipe.id"))
    rcp = relationship(Recipe)

    @property
    def recipe(self):
        return self.rcp.name if self.rcp else None

    @recipe.setter
    def recipe(self, name):
        id = registry.get(name, None)
        if id:
            self.recipe_id = id
            return



        
