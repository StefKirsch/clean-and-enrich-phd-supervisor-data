from pybliometrics.scopus import AuthorSearch
from pybliometrics.scopus import ScopusSearch

def find_publications(last_name, initials):
    # Format the query
    query = f"AU-ID({last_name}, {initials})"

    # Perform the author search
    search_result = AuthorSearch(query)
    publications = []

    # If we find authors matching our query
    if search_result.authors:
        # Assuming the first result is the author we're interested in
        author_id = search_result.authors[0].eid

        # Fetch publications by author ID
        s = ScopusSearch(f"AU-ID({author_id})")

        # Collect publication titles
        publications = [doc.title for doc in s.results]

    return publications


def find_first_publication(last_name, initials):
    publications = find_publications(last_name, initials)
    return publications[0] if publications else None


def common_pub_author_and_contributor_1_row(row):
    # Split names and retrieve publications for both authors
    publications_author = find_publications(*row['author_name'].split(', ')) if pd.notnull(row['author_name']) else []
    publications_contributor = find_publications(*row['contributor_1'].split(', ')) if pd.notnull(row['contributor_1']) else []
    
    # Find the first common publication
    return find_common_publication(publications_author, publications_contributor)


def find_common_publication(publications1, publications2):
    # Convert the lists to sets for faster intersection operation
    set1 = set(publications1)
    set2 = set(publications2)

    # Find the intersection of the two sets
    common_publications = set1.intersection(set2)

    # Return the first common publication, if any
    return list(common_publications)[0] if common_publications else None
