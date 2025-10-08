# practicing accessing data from database

import sqlite3
import pandas as pd

conn = sqlite3.connect('fisheries.db')

df_regions = pd.read_sql_query('SELECT * FROM regions', conn)
print(df_regions)