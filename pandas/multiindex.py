import pandas as pd

ix = pd.MultiIndex(levels=[[33, 34], [2007, 2008, 2009]],
                   codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
                   names=[u'id', u'yr'])
