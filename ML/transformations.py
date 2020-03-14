import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import base

from itertools import chain

import warnings
warnings.filters('ignore')

class ToSupervised(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col, groupcol, numlags, dropna=False):

        self.col = col
        self.groupcol = groupcol
        self.nlags = numlags
        self.dropna = dropna

    def fit(self, X, y=None):
        self.X = X
        return self

    def transform(self, X):
        
        tmp = self.X.copy()
        
        for i in range(1, self.nlags):
            tmp[str(i)+'_Day_'+self.col] = tmp.groupby([self.groupcol])[self.col].shift(i)

        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop=True)

        return tmp


class SuperVisedDiff(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col, groupcol, nlags, dropna=False):

        self.col = col
        self.groupcol = groupcol
        self.nlags = nlags
        self.dropna = dropna

    def fit(self, X, y=None):
        self.X = X
        return self

    def transform(self, X):
        tmp = self.X.copy()

        for i in range(1, self.nlags+1):
            tmp[str(i)+'_Day_'+self.col] = tmp.groupby([self.groupcol])[self.col].diff(i)

        if self.dropna:
            tmp = tmp.dropna()
            tmp = tmp.reset_index(drop=True)

        return tmp

class KFold_Time():

    def __init__(self, **options):

        self.target = options.pop('target', None)
        self.date_col = options.pop('date_col', None)
        self.date_init = options.pop('date_init', None)
        self.date_final = options.pop('date_final', None)

    if options:
        raise TypeError("Invalid paramters passed: {}".format(str(options)))

    if ((self.target) == None) | (self.date_col == None) | (self.date_init == None) | (self.date_final == None):
        raise TypeError("Missing a required input, make sure we have target, date_col, date_init, and date_final inputs.")

    def _train_teste_split_time(self, X):
        n_arrays = len(X)
        
        if n_arrays == 0:
            raise ValueError("Analysis requires at least one array")

        for i in range(self.date_init, self.date_final):

            train = X[X[self.date_col]<i]
            val = X[X[self.date_col]==i]

            X_train, X_test = train.drop([self.target], axis=1), val.drop([self.target], axis=1)
            y_train, y_test = tra[self.target].values, val[self.target].values

            yield X_train, X_test, y_train, y_test

    def split(self, X):
        outs = self._train_teste_split_time(X)
        return chain(outs)

    


