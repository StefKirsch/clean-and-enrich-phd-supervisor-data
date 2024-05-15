import pandas as pd
from pybliometrics.scopus import AuthorSearch
from pybliometrics.scopus import AuthorRetrieval
from src.clean_names_helpers import format_name_to_lastname_initials
from src.clean_names_helpers import format_initials

def find_publications(last_name, initials):
    
    # what I actually need is a function that returns all publications with an author of that name.
    # then I can later easily see if we have a publication with the same eid between two author names 
    # (not actual authors)
    
    # Format the query
    query = f"AUTHLAST({last_name}) and AUTHFIRST({initials})"
    
    print(f"the query is {query}")
    
    # Perform the author search
    search_result = AuthorSearch(query)
    publications = []

    # Filter for exact matches
    exact_matches = []
    # ensure that the formatting is the same between the names
    #original_formatted_name = format_name_to_lastname_initials(f"{last_name}, {initials}")
    
    original_formatted_name = format_initials(f"{last_name}, {initials}")
    
    print(f"the original_formatted_name is {original_formatted_name}")
    
    print(len(search_result.authors))
    
    for author in search_result.authors:
        
        # Format the returned author's name
        formatted_name = format_name_to_lastname_initials(f"{author.surname}, {author.givenname}")
        
        print(f"Checking for exact match with {formatted_name}")
        
        # Check for exact match
        if formatted_name.lower() == original_formatted_name.lower():
            exact_matches.append(author)
            print("Match")

    # If exact matches found, fetch publications for these authors
    if exact_matches:
        # Assuming the first exact match is the author of interest
        # note, every auther should have an eid, while not every author has an Orchid
        # This is not to be confused with the document eid, 
        # scopus keeps an eid for that as well!
        author_eid = exact_matches[0].eid

        # Retrieve the author by eid
        au = AuthorRetrieval(author_eid)
        
        # get a df with the publications
        # `refresh` specifies the number of caching days for the result.
        # After 10 days we pull the results again
        publications = pd.DataFrame(au.get_documents(refresh=10))

    return publications


# def find_first_publication(last_name, initials):
#     publications = find_publications(last_name, initials)
#     return publications[0] if publications else None


def common_pub_author_and_contributor_1_row(row):
    # Split names and retrieve publications for both authors
    
    print(f"The PhD is called: {row['author_name']}")
    print(f"The supervisor is called: {row['contributor_1']}")
    
    row['author_name'].split(', ')
    
    print("Split successful!")
    
    publications_author = find_publications(*row['author_name'].split(', ')) if pd.notnull(row['author_name']) else []
    publications_contributor = find_publications(*row['contributor_1'].split(', ')) if pd.notnull(row['contributor_1']) else []
    
    # eids_author = publications_author['eid']
    # eids_contributor = publications_contributor['eid']
    
    # print(f"The first publication of the PhD is: {eids_author[0]}")
    # print(f"The first publication of the supervisor is: {eids_contributor[0]}")
    
    # common_publications = find_common_publications(eids_author, eids_contributor)
    
    # print(f"The first common publication is {common_publications[0]}")
    
    # Find the first common publication
    return publications_author


def find_common_publications(publications1, publications2):
    # Convert the lists to sets for faster intersection operation
    set1 = set(publications1)
    set2 = set(publications2)

    # Find the intersection of the two sets
    common_publications = set1.intersection(set2)

    # Return common publication, if any
    return list(common_publications) if common_publications else None
