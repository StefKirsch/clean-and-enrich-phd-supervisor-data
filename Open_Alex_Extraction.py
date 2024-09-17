# %% [markdown]
# # Look up publications data and contributors on Open Alex

# %% [markdown]
# ## Load Dependencies

# %%
from pyalex import Works, Authors
import pandas as pd
import numpy as np
import urllib.parse
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% [markdown]
# ## Settings and constants

# %%
# limit the number of rows that are shown with printing dataframes
pd.set_option('display.max_rows', 5)

# %%
# ANSI escape codes for colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'

# Number of rows to read of the full dataset.
NROWS = None # None for all

# %% [markdown]
# ## Reading (and sampling) the data

# %% [markdown]
# `biomedical_pubs.csv` is a subset of the original dataset containing publications __from the last 5 years__ and from __biomedical institutions only__ - amcpub, lumc, vumc, umcu. It contains 1856 records.
# 
# In the code the first two contributors are considered (as potential supervisors) - `contributors_lst(df)` takes a specific numer of columns of a `df` - can be changed if needed.

# %%
pubs_sample = pd.read_csv("data/cleaned/biomedical_pubs.csv")

if NROWS == None:
    n_sample = len(pubs_sample)
else:
    n_sample = NROWS

pubs_sample = pubs_sample.sample(n=n_sample, random_state=40).reset_index(drop=True)

pubs_sample

# %%
# create pairs (author, title) for each row of the df
def author_title_pairs_lst(df):
    tuples_list = []                                             
    for _, row in df.iterrows():
        tuple_values = (row['author_name'], row['title'])
        tuples_list.append(tuple_values)
    return tuples_list

author_title_pairs = author_title_pairs_lst(pubs_sample)

author_title_pairs

# %%
# create lists of contributors for each row of the df
def contributors_lst(df):
    contributors_list = []                                       
    for _, row in df.iloc[:, 7:9].iterrows():
        contributors = []
        for value in row:
            if type(value) != float:
                contributors.append(value)
        contributors_list.append(contributors)
    return contributors_list

contributors_list = contributors_lst(pubs_sample)

# %% [markdown]
# ## Find the author

# %% [markdown]
# #### Get pubs from our dataframe

# %%
def name2initials_comma_surname(row):                                                     # needs a row of the pubs df
    """
    Takes a row from a DataFrame and converts the author's name
    into the format "Initials. Surname".
    Accepts names in both "Surname, Initials" and "Initials Surname" formats, adding spaces
    between initials if necessary. The resulting name is returned in the desired format.

    Args:
        row (pd.Series): A row from the DataFrame containing the author's name.

    Returns:
        str: The formatted author's name in the format "Initials. Surname".
    """
    author = author_title_pairs[row][0]
    if "," not in author:
        surname = author.split(" ")[0]
        initials = author.split(" ")[1].strip()
    else:
        surname = author.split(",")[0]
        initials = author.split(",")[1].strip()
    matched_initials = re.findall(r'\b[A-Z](?:\.[A-Z])\b', initials)   # adding space between initials if there is no
    if len(matched_initials)>0 and len(matched_initials[0])>2:         # essential transformation if authors cannot be matched
        initials_list = list(initials)                                 # with found title
        for initial in initials_list:
            if initial == ".":
                initials_list.remove(initial)
        name = ""
        for initial in initials_list:
            name += initial + ". "
        name += surname
    else:
        name = initials + " " + surname
    return name

def name2initials_surname(row):                                      # name as it is in the pubs df
    """
    Takes a row from a DataFrame and converts the author's name 
    in the format "Initials Surname".
    Accepts names in both "Surname, Initials" and "Initials Surname" formats.

    Args:
        row (pd.Series): A row from the DataFrame containing the author's name.

    Returns:
        str: The formatted author's name in the format "Initials Surname".
    """
    author = author_title_pairs[row][0]
    if "," not in author:
        surname = author.split(" ")[0]
        initials = author.split(" ")[1].strip()
    else:
        surname = author.split(",")[0]
        initials = author.split(",")[1].strip()
    name = initials + " " + surname
    return name

# %%
def title2url(row):
    """
    Takes a row from a DataFrame, extracts the title of the publication, 
    and replaces `&` and `,` characters with an empty string. The modified title is then 
    URL-encoded.

    Args:
        row (pd.Series): A row from the DataFrame containing the title of the publication.

    Returns:
        str: The transformed and URL-encoded title.
    """
    title = author_title_pairs[row][1]
    replace_dict = {'&': '', ',': ''}
    for k,v in replace_dict.items():
        title = title.replace(k,v)
    title = urllib.parse.quote(title)
    return title

# %% [markdown]
# #### Confirm the author

# %% [markdown]
# Create global variables for finding/not finding the author messages

# %%
found_with_title = 0
found_with_title_name_var = 0
found_with_str = 0
found_with_str_var = 0
not_found = 0

# %%
def confirm_author(row):                                                 # match to existing work, if not found, search by name

    # global counters
    global found_with_title
    global found_with_title_name_var # why is this relevant?
    
    author_id = ''

    try:
        # Look up verbatim publication title via API
        query = Works().search_filter(title=title2url(row)).get()

        if query == []:
            return "Title not found in openAlex"    # based on name

        print(f'{len(query)} match(es) for the publication title found.')
        
        # Loop over responses (=title matches) in query
        for response in query:
            found = False
            if response["authorships"] == []:
                print("This title has no author on openAlex")
                
            # Loop over authors that are in the response
            for author in response["authorships"]:
                
                ## rewrite this so that we are only doing two things:
                # 1. Match the actual NARCIS author with OA authors (all display name variants) verbatim
                # 2. Same with both names converted to standardized intials with format_name_to_lastname_initials()
                
                # try to match the work's author from our dataset with the author from open alex (verbatim)
                if author["author"]["display_name"] == name2initials_comma_surname(row):    # first try matching on "raw" name
                    print(MAGENTA + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
                    
                    found_with_title += 1
                    author_id = author["author"]["id"]
                    found = True
                    
                # Alternative matching
                if found == False:                                       # raw name matches based on the exact same string
                    q = Authors()[author["author"]["id"]]                # if not found check with ID
                    if (q["display_name"] == name2initials_comma_surname(row) or name2initials_comma_surname(row) in q["display_name_alternatives"]) or (q["display_name"] == name2initials_surname(row) or name2initials_surname(row) in q["display_name_alternatives"]):
                        print(YELLOW + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
                        found_with_title_name_var += 1
                        author_id = author["author"]["id"]
                        found = True
                    if (q["display_name"].lower() == name2initials_comma_surname(row).lower() or q["display_name"].lower() == name2initials_surname(row).lower()):
                        print(YELLOW + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
                        found_with_title_name_var += 1
                        author_id = author["author"]["id"]
                        found = True
                    if q["display_name_alternatives"] != []:
                        for n in q["display_name_alternatives"]:
                            if (n.lower() == name2initials_comma_surname(row).lower() or n.lower() == get_name_without_spaces(row).lower()):
                                print(YELLOW + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
                                found_with_title_name_var += 1
                                author_id = author["author"]["id"]
                                found = True
                    if found == False:
                        print("Could not match {} to {}".format(name2initials_comma_surname(row), q["display_name_alternatives"]))
                if found:
                    break
            if found:
                break


    except Exception as e:
        print("An unexpected error occurred:", e)


    if author_id != '':
        return author_id
    if author_id == None:
        return "Confirming author not successful"
    else:
        return "Confirming author not successful"

# %%
def title_DOI(row, confirmed_id):                                         # returns DOI of the title if the author is matched
                                                                          # based on that title
    DOIs = []

    if confirmed_id not in ["Confirming author not successful", "Title not matched"]:
        query = Works().search_filter(title=title2url(row)).get()
        for response in query:
            DOIs.append(response["doi"])

    return DOIs

# %%
def get_author_id(row):                                                   # returns final author ID

    confirmed_id = confirm_author(row)

    if confirmed_id not in ["Confirming author not successful", "Title not matched"]:
        return confirmed_id

    query = Authors().search_filter(display_name=name2initials_comma_surname(row)).get()

    if query == []:
        query = Authors().search_filter(display_name=name2initials_surname(row)).get()

    end_message = confirmed_id
    confirmed_id = ''

    print("{} match(es) for the author found".format(len(query)))
    for response in query:
        global found_with_str
        global found_with_str_var
        found = False
        if (response["display_name"] == name2initials_comma_surname(row) or name2initials_comma_surname(row) in response["display_name_alternatives"]) or (response["display_name"] == name2initials_surname(row) or name2initials_surname(row) in response["display_name_alternatives"]):
            print(BLUE + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
            found_with_str += 1
            confirmed_id = response["id"]
            found = True
        else:                                                               # many other cases ! - DOUBLE SURNAMES - NOT CLEAR WAY TO HANDLE (193 on rs 42)
            if name2initials_comma_surname(row).lower() == response["display_name"].lower():
                print(CYAN + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
                found_with_str_var += 1
                confirmed_id = response["id"]
                found = True
            normalized_name1 = ' '.join(sorted(name2initials_comma_surname(row).split()))
            normalized_name2 = ' '.join(sorted(response["display_name"].split()))
            if normalized_name1 == normalized_name2:
                print(CYAN + "PhD candidate found! - {}".format(name2initials_comma_surname(row)) + RESET)
                found_with_str_var += 1
                confirmed_id = response["id"]
                found = True
        if found:
            break

    if confirmed_id != '':
        return confirmed_id
    else:
        global not_found
        not_found += 1
        return "PhD candidate (probably) not in Open Alex database"                 # OR AUTHOR NOT MATCHED !

# %% [markdown]
# Output colors:
# 
# __Magenta__ - author found in OpenAlex authors based on matched title
# 
# __Yellow__ - author found in OpenAlex authors based on matched title, for alternative version of author's name
# 
# __Blue__ - author found in OpenAlex authors
# 
# __Cyan__ - alternative version of author's name (likely the author) found in OpenAlex authors

# %% [markdown]
# ## Get DOIs for the author

# %% [markdown]
# Not every work in Open Alex has a DOI, thus the returned list of DOIs may be shorter than the number of papers where the author was the first author. The DOI of the work that is already in pubs dataframe is also not included in the list - __only new DOIs are returned__.

# %%
def author_DOIs(row):

    author_id = get_author_id(row)
    if author_id == "PhD candidate (probably) not in Open Alex database":
        print(RED + "Cannot be resolved" + RESET)
        return author_id
    initial_doi = title_DOI(row, author_id)

    result = Works().filter(authorships={"author": {"id": author_id}}).get()
    print("Number of papers where the author was credited: {}".format(len(result)))

    DOIs = []

    i = 0
    for w in result:
        if w["authorships"][0]["author"]["id"] == author_id:
            i += 1
            if w["doi"] != None and w["doi"] not in initial_doi:
                DOIs.append(w["doi"])

    print("Number of papers where the PhD candidate was the first author: {}".format(i))

    if DOIs == []:
        return "No DOIs for the PhD candidate found"
    else:
        return DOIs

# %%
author_DOIs(1)

# %% [markdown]
# ## Get DOIs for the contributors

# %% [markdown]
# #### Transform contributors' names

# %%
def contributors_transformed(row):                                             # with space

    contributors = contributors_list[row]
    transformed_names = []

    for contribtor in contributors:
        surname = contribtor.split(",")[0]
        initials = contribtor.split(",")[1].strip()
        matched_initials = re.findall(r'\b[A-Z](?:\.[A-Z])\b', initials)
        if len(matched_initials)>0 and len(matched_initials[0])>2:
            initials_list = list(initials)
            for initial in initials_list:
                if initial == ".":
                    initials_list.remove(initial)
            name = ""
            for initial in initials_list:
                name += initial + ". "
            name += surname
        else:
            name = initials + " " + surname
        transformed_names.append(name)

    return transformed_names

def contributors_original(row):                                                # as originally in the data
                                                                               # will be used when the transformed names won't
    contributors = contributors_list[row]                                      # match
    transformed_names = []

    for contribtor in contributors:
        surname = contribtor.split(",")[0]
        initials = contribtor.split(",")[1].strip()
        name = initials + " " + surname
        transformed_names.append(name)

    return transformed_names

# %% [markdown]
# #### Find contributors (get IDs)

# %% [markdown]
# Create global variables for finding/not finding the author messages

# %%
contributor_found_with_str = 0
contributor_found_with_str_var = 0
contributor_manual_check = 0
contributor_not_found = 0

# %%
def get_contributors_ids(row):                                                 # returns a list of tuples of contributors and
                                                                               # their corresponding confirmed ids
    contributors = contributors_transformed(row)
    original_contributors = contributors_original(row)
    contributors_ids = []

    for contributor, og_contributor in zip(contributors, original_contributors):
        global contributor_found_with_str
        global contributor_found_with_str_var
        global contributor_manual_check
        global contributor_not_found
        manual = 0
        cont_not_found = 0
        print("Matching {}...".format(contributor))
        found = False
        response_list = []
        query = Authors().search_filter(display_name=contributor).get()        # 2 queries everytime could be too comp expens.
        if query == []:                                                        # check the difference?
            query = Authors().search_filter(display_name=og_contributor).get()
        if query == []:
            print(RED + "        Cannot be resolved: " + RESET + "Could not find the author {}".format(contributor))
            cont_not_found = 1
        else:
            print("        Found {} matches for {}".format(len(query), contributor))
            for i, response in enumerate(query):
                print("          {}:".format(i+1))
                if (response["display_name"] == contributor or contributor in response["display_name_alternatives"]) or (response["display_name"] == og_contributor or og_contributor in response["display_name_alternatives"]):
                    print(BLUE + "            Contributor found! - {}".format(contributor) + RESET)
                    manual = 0
                    cont_not_found = 0
                    contributor_found_with_str += 1
                    contributors_ids.append((contributor, response["id"]))
                    found = True
                else:
                    if contributor.lower() == response["display_name"].lower():
                        print(CYAN + "            Contributor found! - {}".format(contributor) + RESET)
                        manual = 0
                        cont_not_found = 0
                        contributor_found_with_str_var += 1
                        contributors_ids.append((contributor, response["id"]))
                        found = True
                if found == False:
                    if response["affiliations"] == []:
                        print(RED + "            No affiliations with institutions - unable to match" + RESET)
                        cont_not_found = 1
                    else:
                        for institution in response["affiliations"]:
                            if institution["institution"]["country_code"] == "NL":
                                print("            {} associated with {}, NL".format(response["display_name"], institution["institution"]["display_name"]))
                                print("            {} maybe associated with {}".format(contributor, response["display_name"]))
                                print(MAGENTA + "            Requires manual check to confirm" + RESET)
                                manual = 1
                                if ("MANUAL CHECK: if {} (target) is {} (found)".format(contributor, response["display_name"]), response["id"]) not in contributors_ids:
                                    response_list.append(("MANUAL CHECK: if {} (target) is {} (found)".format(contributor, response["display_name"]), response["id"]))
                            else:
                                print(RED + "            No NL institution - unlikely to be the match" + RESET)
                                cont_not_found = 1
                if found:
                    break
                if i == len(query)-1:
                    contributors_ids.extend(response_list)
            if manual == 1:
                contributor_manual_check += 1
            if cont_not_found == 1:
                contributor_not_found += 1

    if contributors_ids == []:
        return "None of the contributors is in Open Alex database"
    else:
        return contributors_ids

# %% [markdown]
# ### Get DOIs

# %%
def contributors_DOIs(row):

    ids_list = get_contributors_ids(row)
    confirmed = []
    manual_check = []

    for pair in ids_list:
        if "MANUAL CHECK" in pair[0]:
            if pair not in manual_check:
                manual_check.append(pair)
        else:
            confirmed.append(pair)

    if ids_list == "None of the contributors is in Open Alex database":
        return ids_list


    DOIs_dict = {}

    for author in confirmed:
        author_id = author[1]
        result = Works().filter(authorships={"author": {"id": author_id}}).get()
        print()
        print("Searching works for {}...".format(author[0]))
        print("        Number of papers where the contributor was credited as an author: {}".format(len(result)))

        DOIs = []

        i = 0
        for w in result:
            if w["authorships"][0]["author"]["id"] == author_id:
                i += 1
                if w["doi"] != None:
                    DOIs.append(w["doi"])

        print("        Number of papers where the contributor was the first author: {}".format(i))

        if author[0] not in DOIs_dict:
            DOIs_dict[author[0]] = DOIs

    for author in manual_check:
        author_id = author[1]
        extract_name = re.search(r'if\s+(.*?)\s+is', author[0])
        name = extract_name.group(1)
        result = Works().filter(authorships={"author": {"id": author_id}}).get()
        print()
        print("Searching works for {}... - NOT CONFIRMED".format(name))
        print("        Number of papers where the contributor was credited as an author: {}".format(len(result)))

        DOIs = []

        i = 0
        for w in result:
            if w["authorships"][0]["author"]["id"] == author_id:
                i += 1
                if w["doi"] != None:
                    DOIs.append(w["doi"])

        print("        Number of papers where the contributor was the first author: {}".format(i))

        name = name + " - NOT CONFIRMED"
        if name not in DOIs_dict:
            DOIs_dict[name] = DOIs


    if DOIs_dict == {}:
        return "No DOIs for the associated contributors found"

    got_dois = False
    for value in DOIs_dict.values():
        if value != []:
            got_dois = True
    if got_dois:
        return DOIs_dict
    else:
        return "No DOIs for the associated contributors found"

# %%
# contributors_DOIs(3)

# %% [markdown]
# ## Run on a subset

# %% [markdown]
# Store the output in `dois_df`

# %%
dois_data = []  # empty list to store data for each row of the df

for i in tqdm(range(len(pubs_sample))):  
    contributors = contributors_DOIs(i)  # get the DOIs of contributors for every row (i)

    # check if there are no DOIs for contributors or if none of the contributors are in the Open Alex database
    if contributors == "None of the contributors is in Open Alex database" or contributors == "No DOIs for the associated contributors found":
        message_contributors = contributors  # store the message for why there are no DOIs for contributors
        contributors = np.nan  # set contributors to NaN since no valid DOIs were found
        contr_count = 0  # set the count of contributors with DOIs to 0
        contr_dois_count = 0  # set the cumulative count of DOIs for contributors to 0
    else:
        # calculate the number of contributors with DOIs and the cumulative count of DOIs
        contr_count = sum(1 for lst in contributors.values() if lst)
        contr_dois_count = sum(len(lst) for lst in contributors.values())
        message_contributors = np.nan  # no error message needed since DOIs were found

    author_dois = author_DOIs(i)  # get the DOIs of the author (PhD candidate) for the current row (i)

    # check if there are no DOIs for the author or if the author is not in the Open Alex database
    if author_dois == "PhD candidate (probably) not in Open Alex database" or author_dois == "No DOIs for the PhD candidate found":
        message_author = author_dois  # store the message for why there are no DOIs for the author
        author_dois = np.nan  # set author DOIs to NaN since no valid DOIs were found
        count = 0  # set the count of DOIs for the author to 0
    else:
        count = len(author_dois)  # set the count of DOIs for the author
        message_author = np.nan  # no error message needed since DOIs were found

    # create a dictionary with all the relevant information for the current row
    data = {
        'PhD candidate': name2initials_surname(i),  
        'PhD candidate DOIs found in OpenAlex': author_dois,  
        'DOIs count': count,  
        'Contributors-DOIs Dictionary': contributors, 
        'Number of contributors with DOIs found in OpenAlex': contr_count, 
        'Cumulative found Contributor DOIs count': contr_dois_count,
        'Why no DOIs for PhD candidate': message_author,  
        'Why no DOIs for contributors': message_contributors 
    }
    dois_data.append(data)  # append the dictionary to the list of data

dois_df = pd.DataFrame(dois_data)  # convert the list of dictionaries into a DataFrame

# %%
dois_df = pd.DataFrame(dois_data)
dois_df

# %%
dois_df.to_csv('data/output/results.csv', index=False) 

# %% [markdown]
# Statistics for how and if PhD candidates / contributors were found:

# %%
phd_cand_verification = {
    "Number of PhD candidates found with matched title": found_with_title,
    "Number of PhD candidates found with matched title, but name variation": found_with_title_name_var,
    "Number of PhD candidates found with name string": found_with_str,
    "Number of PhD candidates found with name string variation": found_with_str_var,
    "Number of not found PhD candidates": not_found
}

contributor_verification = {
    "Number of contributors found with name string": contributor_found_with_str,
    "Number of contributors found with name string variation": contributor_found_with_str_var,
    "Number of contributors for manual check": contributor_manual_check,
    "Number of not found contributors": contributor_not_found
}

df_pnd_cand_ver = pd.DataFrame(list(phd_cand_verification.items()), columns=["Found how?", "Value"])
df_contributor_ver = pd.DataFrame(list(contributor_verification.items()), columns=["Found how?", "Value"])

# %%
df_pnd_cand_ver

# %%
df_contributor_ver

# %% [markdown]
# ## Visualisations & Statistics

# %%
author_doi_counts = dois_df.groupby('PhD candidate')['DOIs count'].sum()

author_doi_counts = author_doi_counts.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
author_doi_counts.plot(kind='bar', color='rosybrown')
plt.title('Number of DOIs per PhD candidate (Sorted) - found in Open Alex')
plt.ylabel('Number of DOIs')
plt.locator_params(axis='y', integer=True)
plt.xticks([])
plt.tight_layout()
plt.show()

# %%
author_contributors_count = dois_df.groupby('PhD candidate')['Number of contributors with DOIs found in OpenAlex'].sum()

author_contributors_count = author_contributors_count.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
author_contributors_count.plot(kind='bar', color='pink')
plt.title('Number of Contributors per PhD candidate (Sorted) - found in Open Alex')
plt.ylabel('Number of Contributors')
plt.locator_params(axis='y', integer=True)
plt.xticks([])
plt.tight_layout()
plt.show()

# %%
author_cumulative_dois_count = dois_df.groupby('PhD candidate')['Cumulative found Contributor DOIs count'].sum()

author_cumulative_dois_count = author_cumulative_dois_count.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
author_cumulative_dois_count.plot(kind='bar', color='tan')
plt.title('Cumulative Number of Contributor DOIs Count per PhD candidate (Sorted)')
plt.ylabel('Cumulative Number of DOIs')

plt.locator_params(axis='y', integer=True)

plt.xticks([])
plt.tight_layout()
plt.show()

# %%
mean_dois_per_author = dois_df['DOIs count'].mean()
median_dois_per_author = dois_df['DOIs count'].median()

mean_found_contributors = dois_df['Number of contributors with DOIs found in OpenAlex'].mean()
median_found_contributors = dois_df['Number of contributors with DOIs found in OpenAlex'].median()

mean_cumulative_dois_count = dois_df['Cumulative found Contributor DOIs count'].mean()
median_cumulative_dois_count = dois_df['Cumulative found Contributor DOIs count'].median()

nan_author_dois_count = dois_df['PhD candidate DOIs found in OpenAlex'].isnull().sum()
nan_contributors_dois_count = dois_df['Contributors-DOIs Dictionary'].isnull().sum()

summary_data = {
    'Statistics': ['Mean DOIs per PhD candidate', 'Median DOIs per PhD candidate',
                   'Mean Found Contributors', 'Median Found Contributors',
                   'Mean Cumulative DOIs Count', 'Median Cumulative DOIs Count',
                   'NaN Values for PhD candidate DOIs', 'NaN Values for Contributors DOIs'],
    'Values': [mean_dois_per_author, median_dois_per_author,                           # round if needed
               mean_found_contributors, median_found_contributors,
               mean_cumulative_dois_count, median_cumulative_dois_count,
               nan_author_dois_count, nan_contributors_dois_count]
}

summary_df = pd.DataFrame(summary_data)

summary_df

# %%
dois_df.fillna('Unknown', inplace=True)

unique_messages_column1 = dois_df['Why no DOIs for PhD candidate'].value_counts()
unique_messages_column2 = dois_df['Why no DOIs for contributors'].value_counts()

unique_messages_column1 = unique_messages_column1.drop('Unknown', errors='ignore')
unique_messages_column2 = unique_messages_column2.drop('Unknown', errors='ignore')

plt.figure(figsize=(11, 7))

plt.bar(unique_messages_column1.index, unique_messages_column1.values, alpha=0.5, label='Why no DOIs for PhD candidate', color='lightcoral')
plt.bar(unique_messages_column2.index, unique_messages_column2.values, alpha=0.5, label='Why no DOIs for contributors', color='burlywood')

plt.ylabel('Count')
plt.title('Why DOIs could not be extracted')
plt.xticks(rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Filter for the task at hand

# %%
df_clean = dois_df.iloc[:, :6].dropna()
# df_clean.reset_index(drop=True, inplace=True)
df_clean

# %%
mean_dois_per_author = df_clean['DOIs count'].mean()
median_dois_per_author = df_clean['DOIs count'].median()

mean_found_contributors = df_clean['Number of contributors with DOIs found in OpenAlex'].mean()
median_found_contributors = df_clean['Number of contributors with DOIs found in OpenAlex'].median()

mean_cumulative_dois_count = df_clean['Cumulative found Contributor DOIs count'].mean()
median_cumulative_dois_count = df_clean['Cumulative found Contributor DOIs count'].median()

summary_data = {
    'Statistics': ['Mean DOIs per PhD candidate', 'Median DOIs per PhD candidate',
                   'Mean Found Contributors', 'Median Found Contributors',
                   'Mean Cumulative DOIs Count', 'Median Cumulative DOIs Count'],
    'Values': [mean_dois_per_author, median_dois_per_author,
               mean_found_contributors, median_found_contributors,
               mean_cumulative_dois_count, median_cumulative_dois_count]
}

summary_df = pd.DataFrame(summary_data)

summary_df

# %%
# df_clean.to_csv('complete_extraction.csv', index=False)


