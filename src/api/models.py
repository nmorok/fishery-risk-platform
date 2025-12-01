'''
take the data and build linear models
'''

import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.linear_model import LinearRegression

DB_PATH = 'data/fishery_disasters.db'
CSV_DIR = Path('data/csv')



# build the linear model
model = LinearRegression()
model.fit()
