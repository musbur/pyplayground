from functools import lru_cache, wraps


class CachedDB():
    @lru_cache()
    def expensive(self, par):
        print('CachedDB', self.db, par)
        return par

    def get(self, db, par):
        self.db = db
        return self.expensive(par)

class Cloak():
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True

@lru_cache()
def cloaked(cloak, par):
    print('Cloak', cloak.db, par)
    return par

def lru_ignore_first(func):
    '''No idea how to implement this. Is it even possible?'''
    pass

@lru_ignore_first
def decorated(db, par):
    print('with decorator', par)
    return

cached_db = CachedDB()
cloak = Cloak()
for i in range(5):
    r = cached_db.get(i, 100)
    print(r)
    cloak.db = i
    r = cloaked(cloak, 200)
    print(r)
#    r = decorated(i, 100)
#    print(r)

