# %% [markdown]
# # Subset dataset to publications from Medical institutions only 
# 
# ## Load dependencies

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# %% [markdown]
# ## Settings

# %%
# limit the number of rows that are shown with printing dataframes
pd.set_option('display.max_rows', 5)

# %%
pubs = pd.read_csv("data/cleaned/pubs.csv")

pubs

# %% [markdown]
# ### Filter for biomedical institutions

# %%
biomedical = ['amcpub', 'lumc', 'vumc', 'umcu'] 
pubs_biomed = pubs[pubs['institution'].isin(biomedical)].reset_index(drop=True)

pubs_biomed

# %% [markdown]
# ### Filter out recent publications
# 
# We don't want to go too far back in time, where reporting might be too spotty to have reliable data. So we only go back 5 years from the reporting date and filter the dataset by that.

# %%
# define reporting date. We include publications that are at least 5 years old from this date.

# checked if using June, July or August 1st makes a difference and it does not. 
# We get still include the same amount of publications <3
date_str = "01-07-2024"
reporting_date = datetime.strptime(date_str, "%d-%m-%Y")

# convert `year` column to datetime so we can use datetime operations on it
pubs_biomed['year'] = pd.to_datetime(pubs_biomed['year'], format='%Y')
    
date_5_years_ago = reporting_date - timedelta(days=5*365)
pubs_recent = pubs_biomed[pubs_biomed['year'] >= date_5_years_ago].reset_index(drop=True)

pubs_recent

# %% [markdown]
# ### Additional cleaning

# %% [markdown]
# Replace values that start with illegal strings with `nan` 

# %%
illegal_startswith = [',', 'Surgery'] # unclear why we should remove Surgery. But keep it in the early August release

def illegal_startswith2nan(value):
    value_str = str(value)
    if any(value_str.startswith(word) for word in illegal_startswith):
        return np.nan
    return value

pubs_wo_illegal_startswith = pubs_recent.map(illegal_startswith2nan)

pubs_wo_illegal_startswith

# %%
# identify rows where ALL contributor rows are nan
contributor_nan_rows = pubs_wo_illegal_startswith.filter(like='contributor').isnull().all(axis=1)

# ... and remove them
pups_has_contrib = pubs_wo_illegal_startswith[~contributor_nan_rows].reset_index(drop=True)

pups_has_contrib

# %% [markdown]
# Make sure every publication has a continious list of contributors without nan gaps. To do this, promote contributors that have a `NaN` contributor higher to them, so that there are no `NaN` contributors higher than non-`NaN` contributors.

# %%
pups_continious_contrib = pups_has_contrib

contributor_columns = [col for col in pups_continious_contrib.columns if col.startswith('contributor_')]

for index, row in pups_continious_contrib.iterrows():
    for i in range(1, len(contributor_columns)-1):
        current_col = 'contributor_' + str(i)
        next_col = 'contributor_' + str(i + 1)
        
        # Check if current column is NaN and next column is not NaN
        if pd.isna(row[current_col]) and not pd.isna(row[next_col]):
            print(row)
            # Replace current column with next column and set next column to NaN
            pups_continious_contrib.at[index, current_col] = row[next_col]
            pups_continious_contrib.at[index, next_col] = np.nan
            
            
pups_continious_contrib

# %% [markdown]
# ### Drop NaN rows and duplicates

# %%
pups_cleaned = pups_continious_contrib
pups_cleaned = pups_cleaned.dropna(subset=['author_name', 'title', 'contributor_1'])

pups_cleaned = pups_cleaned.drop_duplicates(subset=['author_name'], keep='last').reset_index(drop = True)

pups_cleaned

# %% [markdown]
# ### Save to CSV - USE IN `Open_Alex_Final.ipynb`

# %%
pups_cleaned.to_csv('data/cleaned/biomedical_pubs.csv', index=False)


