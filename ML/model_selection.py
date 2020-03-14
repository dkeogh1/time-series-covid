import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import base

import warnings
warnings.filters('ignore')

class ToSupervised(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col, groupcol, numlags, dropna=False)

        self.col = col
        self.groupcol = groupcol
        self.numlags = numlags
        self.dropna = dropna

    def fit(self, X, y=None):
        self.X = X
        return self

    def transform(self, X):
        
        tmp = self.X.copy()
        
        for i in range(1, self.numlags):
            tmp[str(i)+'_Day'+'_'+self.col] = tmp.groupby([self.groupcol])[self.col].shift(i)

        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop=True)

        return tmp


