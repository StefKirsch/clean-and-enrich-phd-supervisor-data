import pandas as pd
from nameparser import HumanName
import spacy
from utils.clean_names import remove_non_person_contributors_and_export

# Load the multilingual NER model
nlp = spacy.load("xx_ent_wiki_sm")

def format_name_to_lastname_initials(name):
    human_name = HumanName(name)
    # Extract the last name
    last_name = human_name.last
    # Extract the first name and/or initials
    first_names = human_name.first + ' ' + human_name.middle
    # Convert first names to initials, keeping existing initials as is
    initials = ''.join([f"{name[0]}." if name else '' for name in first_names.split()])
    # Combine last name and initials
    formatted_name = f"{last_name}, {initials}" if initials else last_name
    return formatted_name.strip()

nrows = 100
removed_contributors = []

pairs_raw = pd.read_csv("../data/raw/pairs_sups_phds.csv", nrows=nrows)
pairs_raw = pairs_raw.convert_dtypes() # make sure all integer columns are integer dtype 

# remove duplicates
pairs_filtered = pairs_raw.drop_duplicates() 
# remove contributors that aren't people
csv_path = "../data/removed_contributors.csv"
pairs_filtered = remove_non_person_contributors_and_export(pairs_filtered, csv_path, nlp)

# Standardize names
pairs_std = pairs_filtered
# Apply name standardization to the contributor column
pairs_std['contributor'] = pairs_filtered['contributor'].apply(format_name_to_lastname_initials)

pairs_std.head()

# Group by publication and aggregate contributors into a list
aggregated = pairs_std.groupby(['integer_id', 'thesis_identifier', 'institution', 'author_name', 'title', 'year', 'language']) \
                  .agg(list) \
                  .reset_index()

# Initialize a list to hold publication data dictionaries
pubs_list = []

# Iterate over each aggregated group
for _, row in aggregated.iterrows():
    # Initialize a dictionary with publication information
    pub_dict = {col: row[col] for col in ['integer_id', 'thesis_identifier', 'institution', 'author_name', 'title', 'year', 'language']}
    
    # Get the list of contributors and their orders for this publication
    contributors = row['contributor']
    contributor_orders = row['contributor_order']
    
    # Add contributors to the dictionary using dynamic keys
    for order in sorted(set(contributor_orders)):  # Ensure unique and sorted order numbers
        if order - 1 < len(contributors):  # Check to prevent index error
            pub_dict[f'contributor_{order}'] = contributors[order - 1]
    
    # Append the publication dictionary to the list
    pubs_list.append(pub_dict)

# Convert the list of dictionaries to a DataFrame
pubs = pd.DataFrame(pubs_list)

# Ensure correct data types and fill missing values with a suitable placeholder if necessary
pubs = pubs.convert_dtypes()

# Display the transformed DataFrame
pubs.head()