import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import base

import warnings
warnings.filterwarnings('ignore')

class BaseEstimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self, predcol)

        self.predcol = predcol

    def fit(self, X, y)
        return self
    
    def predict(self, X):
        prediction = X[self.predcol].values
        return prediction

    def score(self, X, y, scoring)
        prediction = self.predict(X)

        error = scoring(y, prediction)

        return error


class TimeSeriesRegressor(base.BaseEstimator, base.RegressorMixin):

    def __init__(self, model, cv, scoring, verbosity=True):
        self.model = model,
        self.cv = cv,
        self.verbosity = verbosity,
        self.scoring = scoring

    def fit(self,X,y=None):
        return self
    
    def predict(self,X=None)

        pred = {}
        for index, fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold
            self.model.fit(X_train, y_train)
            pred[str(index)+'_fold'] = self.model.predict(X_test)

        prediction = pd.DataFrame(pred)

        return prediction

    
    def score(self, X, y=None):

        errors = []
        for index, fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, y_train)
            prediction = self.model.predict(X_test)
            error = self.scoring(y_test, prediction)
            errors.append(error)

            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(index,error))

        if self.verbosity:
            print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors


class TimeSeriesRegressorLog(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self,model,cv,scoring,verbosity=True):
        self.model = model
        self.cv = cv
        self.verbosity = verbosity
        self.scoring = scoring
        
            
    def fit(self,X,y=None):
        return self
        
    
    def predict(self,X=None):
        
        pred = {}
        for indx,fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, y_train)
            pred[str(indx)+'_fold'] = self.model.predict(X_test)
            
        prediction = pd.DataFrame(pred)
    
        return prediction

    
    def score(self,X,y=None):#**options):


        errors = []
        for indx,fold in enumerate(self.cv.split(X)):

            X_train, X_test, y_train, y_test = fold    
            self.model.fit(X_train, np.log1p(y_train))
            prediction = np.expm1(self.model.predict(X_test))
            error = self.scoring(y_test, prediction)
            errors.append(error)

            if self.verbosity:
                print("Fold: {}, Error: {:.4f}".format(indx,error))

        if self.verbosity:
                print('Total Error {:.4f}'.format(np.mean(errors)))

        return errors