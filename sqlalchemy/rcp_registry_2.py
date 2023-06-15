from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.associationproxy import association_proxy

Base = declarative_base()


class SeqRecipeRegistry():
    def __init__(self, db):
        self.db = db
        self.data = dict()

    def get(self, name):
        recipe = self.data.get(name)
        if not recipe:
            recipe = (self.db.query(SeqRecipe)
                      .filter_by(name=name)
                      .one_or_none())
            if not recipe:
                recipe = SeqRecipe(name=name)
            self.data[name] = recipe
        return recipe

class SeqRecipe(Base):
    __tablename__ = "recipe"
    id            = Column(Integer, primary_key=True)
    name          = Column(String(200), unique=True)

def create_recipe(name):
    return _recipe_registry.get(name)

def gsf(a, b):

    def getter(recipe):
        return recipe.name

    def setter(recipe, name):
        raise UserWarning("Readonly")

    return getter, setter

class Sequence(Base):
    __tablename__ = "sequence"
    id            = Column(Integer, primary_key=True)
    recipe_id     = Column(ForeignKey("recipe.id"))
    recipe = relationship(SeqRecipe)

    recipe_name = association_proxy("recipe", "name",
                                    creator=create_recipe,
                                    getset_factory=gsf)

engine = create_engine("sqlite:///test.db")
Base.metadata.create_all(engine)

db = sessionmaker(engine)()

_recipe_registry = SeqRecipeRegistry(db)

for n in ('Meep', 'Maap', 'Moop', 'Meep'):
    r = Sequence(recipe_name=n)
    print(r.recipe_name)
    db.add(r)
db.commit()

rs = (db.query(Sequence).filter_by(recipe_name="Meep"))
for r in rs:
    print(r)
