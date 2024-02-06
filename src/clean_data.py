import pandas as pd

nrows = 100

pairs = pd.read_csv("../data/raw/pairs_sups_phds.csv", nrows=nrows)
pairs = pairs.convert_dtypes()

pairs.head()

