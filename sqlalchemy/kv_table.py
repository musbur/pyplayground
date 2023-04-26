from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class RecipeRegistry():
    def __init__(self, db):
        self.db = db
        self.data = dict()

    def get(self, name):
        recipe = self.data.get(name)
        if not recipe:
            recipe = (self.db.query(Recipe)
                      .filter_by(name=name)
                      .one_or_none())
            if not recipe:
                recipe = Recipe(name=name)
            self.data[name] = recipe
        return recipe

class Recipe(Base):
    __tablename__ = "recipe"
    id            = Column(Integer, primary_key=True)
    name          = Column(String(200), unique=True)

class Sequence(Base):
    __tablename__ = "sequence"
    id            = Column(Integer, primary_key=True)
    recipe_id     = Column(ForeignKey("recipe.id"))
    _recipe = relationship(Recipe)

    @hybrid_property
    def recipe(self):
        if not self._recipe:
            raise RuntimeError("This shouldn't happen")
        return self._recipe.name

    @recipe.setter
    def recipe(self, name):
        self._recipe = _recipe_registry.get(name)

    @recipe.expression
    def recipe(cls):
        return Recipe.name

engine = create_engine("sqlite:///test.db")
Base.metadata.create_all(engine)

db = sessionmaker(engine)()

_recipe_registry = RecipeRegistry(db)

for n in ('Meep', 'Maap', 'Moop', 'Meep'):
    r = Sequence(recipe=n)
    db.add(r)
db.commit()

rs = (db.query(Sequence)
      .join(Recipe) # I want to get rid of this join
      .filter(Sequence.recipe == "Meep"))
for r in rs:
    print(r)
