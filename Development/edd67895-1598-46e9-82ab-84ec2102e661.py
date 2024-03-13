import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

df = pd.read_csv("lending.csv")
