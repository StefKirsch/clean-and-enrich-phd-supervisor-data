import logging
from pyalex import Authors, Works
import pandas as pd
from os import path, makedirs

from src.io_helpers import fetch_supervisors_from_pilot_dataset
from src.clean_names_helpers import format_name_to_lastname_firstname

class AuthorRelations:
    # Class attribute shared by all instances
    # Can be overwritten at the class level to change the default for all (upcoming) instances in scope
    # Keys: supervisor name 
    # Values: supervisor OpenAlex ID
    supervisors_in_pilot_dataset = dict()
    
    def __init__(self, phd_name, title, year, institution, contributors, years_tolerance=0, verbosity='INFO'):
        self.phd_name = phd_name
        self.title = title
        self.year = year
        self.institution = institution
        self.contributors = contributors
        self.years_tolerance = years_tolerance
        self.phd_candidate = None
        self.potential_supervisors = []
        self.verbosity = verbosity.upper()
        
        # Define target years as a property of the object
        self.target_years = self.calculate_target_years()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def calculate_target_years(self):
        """
        Calculates the target years based on the years_tolerance.
        If years_tolerance is negative, includes years before self.year.
        If years_tolerance is positive, includes years after self.year.
        """
        if self.years_tolerance == 0:
            return [self.year]
        elif self.years_tolerance > 0:
            return set(range(self.year, self.year + self.years_tolerance + 1))
        else:  # years_tolerance < 0
            return set(range(self.year + self.years_tolerance, self.year + 1))
        
    def setup_logging(self):
        # Map verbosity levels to logging levels
        verbosity_levels = {
            'NONE': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }
        log_level = verbosity_levels.get(self.verbosity, logging.INFO)
        self.logger.setLevel(log_level)

        # Remove all handlers associated with the logger
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        # Set propagate to False to prevent messages from being printed to the console
        self.logger.propagate = False

        # Create a file handler with UTF-8 encoding
        file_handler = logging.FileHandler('author_relations.log', encoding='utf-8')
        file_handler.setLevel(log_level)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

    def search_phd_candidate(self, criteria):
        """
        Search for the PhD candidate by name and validate based on criteria.
        Criteria options: 'affiliation', 'title', 'both'
        """
        self.logger.info(f"Searching for PhD candidate: {self.phd_name}")
        # Search for candidates by PhD name
        candidates = Authors().search(self.phd_name).get()
        self.logger.debug(f"Found: {len(candidates)} people who are potential matches.")

        # If no candidates are found, log and return
        if not candidates:
            self.logger.warning("No candidates found with the given PhD name.")
            return None

        # Filter candidates based on the specified criteria
        for candidate in candidates:
            self.logger.debug(f"Evaluating candidate: {candidate['display_name']} (ID: {candidate['id']})")
            affiliation_match = self.check_affiliation(candidate)
            title_match = self.check_authored_work(candidate)

            match_type = None

            if affiliation_match and title_match:
                match_type = 'affiliation and title'
            elif criteria == ('affiliation' or 'either') and affiliation_match:
                match_type = 'affiliation'
            elif criteria == ('title' or 'either') and title_match:
                match_type = 'title'     

            if match_type:
                self.phd_candidate = candidate
                self.logger.info(f"PhD candidate confirmed by {match_type}: {candidate['display_name']}")
                break
            else:
                self.logger.debug(f"No match found for criteria: {criteria}. Moving to next candidate.")

        if not self.phd_candidate:
            self.logger.warning("PhD candidate not found or criteria not met.")
            return None
        else:
            return self.phd_candidate

    def check_affiliation(self, candidate):
        """
        Compare the affiliation of an candidate to `self.institution` in the target years.
        Return True if it matches and False otherwise.
        """
        affiliations = candidate.get('affiliations', [])
        match_found = False

        if self.verbosity == 'DEBUG':
            self.logger.debug(f"Target Institution: '{self.institution}', Target Years: {self.target_years}")

        for affiliation in affiliations:
            institution_name = affiliation['institution']['display_name']
            years = affiliation.get('years', [])
            is_match = (self.institution == institution_name) and any(year in self.target_years for year in years)
            self.logger.debug(
                f"Checking affiliation: Candidate Institution '{institution_name}', Years: {years} - "
                f"Match Found: {'Yes' if is_match else 'No'}"
            )
            if is_match:
                match_found = True
                break  # Stop checking after a match is found

        if not match_found:
            self.logger.debug("No affiliation match found for this candidate.")

        return match_found

    def get_candidate_affiliations(self, candidate, in_target_years=True):
        """
        Returns a set of institution names that the candidate was affiliated with.

        Parameters:
            candidate (dict): The candidate author object containing affiliation data.
            in_target_years (bool): If True, only include affiliations within the target years.
                                    If False, include all affiliations regardless of year.

        Returns:
            set: A set of institution names affiliated with the candidate.
        """
        affiliations = candidate.get('affiliations', [])
        institutions = set()
        for affiliation in affiliations:
            institution_name = affiliation['institution']['display_name']
            affiliation_years = affiliation.get('years', [])
            
            if not in_target_years or self.target_years.intersection(affiliation_years):
                institutions.add(institution_name)
        return institutions
    
    def check_authored_work(self, candidate):
        """
        Check if the candidate has authored the specified title.
        """
        candidate_id = candidate['id']
        if self.verbosity == 'DEBUG':
            self.logger.debug(f"Target Title: '{self.title}'")

        works = Works().filter(author={"id": candidate_id}).get()
        match_found = False

        for work in works:
            work_title = work['title']
            if not work_title:
                continue
            is_match = self.title.lower() == work_title.lower()
            self.logger.debug(
                f"Checking work: Candidate Work Title '{work_title}' - Match Found: {'Yes' if is_match else 'No'}"
            )
            if is_match:
                match_found = True
                break  # Stop checking after a match is found

        if not match_found:
            self.logger.debug("No title match found for this candidate.")

        return match_found

    def collect_supervision_metadata(self):
        """
        Based on relationships between `self.phd_candidate` and contributors collect metadata that indicates supervision. 
        
        We look up contributors in OpenAlex. We confirm that we matched the contributor name with an author in OpenAlex
        by requiring at least *one* shared affiliation in the time window (target years) of the publication of the thesis.  
        
        Every confirmed contributor is added to the final dataset, the only difference for them will be the metadata that we collect here
        
        The following metadata will be collected per confirmed contributor:
        'is_first': contributor is mentioned as first contributor in underlying dataset -> bool
        'grad_inst': Phd and contributor shared institution in the time window of the thesis publication -> bool
        'n_shared_inst_grad': Number of institutions PhD candidate and contributor share at the time of graduation -> int
        'is_sup_in_pilot_dataset': Contributor is mentioned in the pilot dataset -> bool
        """
        
        if not self.phd_candidate:
            self.logger.warning("PhD candidate not confirmed. Cannot find potential supervisors.")
            return []
        
        # Get PhD candidate's affiliations
        phd_affiliations_at_graduation = self.get_candidate_affiliations(self.phd_candidate, in_target_years=True)
        
        # If the PhD candidate has no affiliations in the target years, return an empty list
        if not phd_affiliations_at_graduation:
            self.logger.warning("PhD candidate has no affiliations in target years. Cannot find potential supervisors.")
            return []
        
        # Log the target institutions (affiliations of the PhD candidate in the target years)
        self.logger.debug(f"Target Institutions: {phd_affiliations_at_graduation}, Target Years: {self.target_years}")
        self.logger.info("Searching for potential supervisors among contributors.")
        
        # Initialize list to store supervisor data
        self.potential_supervisors = []

        for idx, contributor_name in enumerate(self.contributors):
            self.logger.debug(f"Searching for contributor: {contributor_name}")

            # set flag for candidate being the first contributor in the underlying dataset
            is_first = (idx == 0)
            
            # Search for contributors in OpenAlex
            openalex_candidates = Authors().search(contributor_name).get()
            self.logger.debug(f"Found: {len(openalex_candidates)} candidates for contributor '{contributor_name}'.")
            
            # If no candidates are found continue to next contributor
            if not openalex_candidates:
                self.logger.debug(f"No candidates found for contributor: {contributor_name}")
                continue
            
            # Prepare contributor_found_in_openalex flag
            contributor_found_in_openalex = False
            
            for candidate in openalex_candidates:
                # Get all affiliations of supervisor
                supervisor_affiliations = self.get_candidate_affiliations(candidate, in_target_years=True)
                
                # Check for shared affiliations
                shared_affiliations = phd_affiliations_at_graduation.intersection(supervisor_affiliations)
                n_shared_inst_grad = len(shared_affiliations)
                same_grad_inst = self.institution in shared_affiliations

                # Logging per institution we are checking
                for institution in supervisor_affiliations:
                    is_match = institution in phd_affiliations_at_graduation
                    self.logger.debug(
                        f"Checking affiliation: Potential Supervisor '{candidate['display_name']}' Institution '{institution}' - "
                        f"Match Found: {'Yes' if is_match else 'No'}"
                    )
                
                # Flag if the candidate supervisor is in the pilot dataset from class attribute dict
                is_sup_in_pilot_dataset = candidate['id'] in self.__class__.supervisors_in_pilot_dataset.values()
                
                if shared_affiliations:
                    # Collect supervisor data
                    supervisor_data = {
                        'supervisor': candidate,
                        'is_first': is_first,
                        'same_grad_inst': same_grad_inst,
                        'n_shared_inst_grad': n_shared_inst_grad,
                        'is_sup_in_pilot_dataset': is_sup_in_pilot_dataset
                    }
                    self.potential_supervisors.append(supervisor_data)
                    self.logger.info(f"Potential supervisor found: {candidate['display_name']} with shared institutions {shared_affiliations}")
                    contributor_found_in_openalex = True
                    break  # The first match with the correct name is most likely the correct contributor

            if not contributor_found_in_openalex:
                self.logger.debug(f"No shared affiliations found for contributor: {contributor_name}")

        # Log the total number of contributors with matches
        self.logger.info(
            f"Total contributors with shared affiliations: {len(self.potential_supervisors)} out of {len(self.contributors)}"
        )

        if not self.potential_supervisors:
            self.logger.warning("No potential supervisors found.")
        return self.potential_supervisors

    def get_results(self):
        """
        Return a DataFrame with the results.
        """
        if not self.phd_candidate:
            self.logger.warning("No results to return; PhD candidate was not found.")
            return None
        phd_id = self.phd_candidate['id']
        phd_name = self.phd_candidate['display_name']

        # Create a list of dictionaries for each supervisor
        results_list = []
        for supervisor_data in self.potential_supervisors:
            supervisor = supervisor_data['supervisor']
            contributor_name = supervisor['display_name']
            contributor_id = supervisor['id']
            is_first = supervisor_data['is_first']
            same_grad_inst = supervisor_data['same_grad_inst']
            n_shared_inst_grad = supervisor_data['n_shared_inst_grad']
            is_sup_in_pilot_dataset = supervisor_data['is_sup_in_pilot_dataset']

            result_row = {
                'phd_name': phd_name,
                'phd_id': phd_id,
                'contributor_name': contributor_name,
                'contributor_id': contributor_id,
                'phd_name': phd_name,
                'is_first': is_first,
                'same_grad_inst': same_grad_inst,
                'n_shared_inst_grad': n_shared_inst_grad,
                'is_sup_in_pilot_dataset': is_sup_in_pilot_dataset
            }
            results_list.append(result_row)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)

        return results_df

    
def find_phd_and_supervisors_in_row(row):
    """
    Finds author relations information from a DataFrame row.

    Processes the row to find the PhD candidate and potential supervisors,
    and returns a DataFrame per row with the required columns.

    Parameters:
        row (pd.Series): A row from the DataFrame containing publication data.

    Returns:
        pd.DataFrame: A DataFrame with columns as specified.
    """
    # Extract necessary fields
    phd_name = row['phd_name']
    title = row['title']
    year = int(row['year'])
    institution = row['institution']
    contributors = [row[f'contributor_{i}'] for i in range(1, 11) if pd.notna(row.get(f'contributor_{i}', None))]
    
    # Create an instance of AuthorRelations
    author_relations = AuthorRelations(
        phd_name=phd_name,
        title=title,
        year=year,
        institution=institution,
        contributors=contributors,
        years_tolerance=-1,  # Adjust as needed
        verbosity='DEBUG'  # Set to 'NONE' for production
    )
    
    # Search for the PhD candidate using the desired criteria
    author_relations.search_phd_candidate(criteria='either')
    
    # Find potential supervisors among the contributors
    author_relations.collect_supervision_metadata()
    
    # Get the DataFrame results
    results_df = author_relations.get_results()
    
    return results_df
    

def fetch_author_openalex_names_ids(author: str) -> dict[str, str]:
    """
    Looks up a single author in OpenAlex and retrieves all matches as a dictionary.

    Parameters:
        author (str): The name of the author to search in OpenAlex.

    Returns:
        dict: A dictionary where keys are display names and values are OpenAlex IDs.
    """
    try:
        search_results = Authors().search(author).get()

        # Process all matches into a dictionary
        return {
            result['display_name']: result['id']
            for result in search_results
        }
    except Exception as e:
        print(f"Error fetching data for author '{author}': {e}")
        return {}


def get_supervisors_openalex_ids(repo_url, csv_path):
    """
    Retrieves supervisor data with OpenAlex IDs, either by reading from a CSV file or querying OpenAlex.

    Parameters:
        repo_url (str): The URL of the GitHub directory containing supervisor data.
        csv_path (str): Path to the CSV file where supervisor data is stored.

    Returns:
        dict: A dictionary where keys are supervisor names and values are OpenAlex IDs.
    """
    # If the CSV file exists, load it
    if path.exists(csv_path):
        print(f"Loading supervisor data from {csv_path}...")
        supervisors_df = pd.read_csv(csv_path)
        return dict(zip(supervisors_df['supervisor_name'], supervisors_df['supervisor_id']))
    
    # If the CSV file does not exist, fetch data and save it
    print(f"No existing CSV found at {csv_path}. Querying OpenAlex...")
    
    # Fetch the unique supervisors from the dataset
    supervisors = fetch_supervisors_from_pilot_dataset(
        repo_url=repo_url,
        file_extension=".xlsx",
        verbosity=True
    )
    
    # Apply name standardization
    supervisors_std = [format_name_to_lastname_firstname(name) for name in supervisors]
    
    # Query OpenAlex for each supervisor and build the dictionary
    supervisors_ids = {
        display_name: openalex_id
        for supervisor in supervisors_std
        for display_name, openalex_id in fetch_author_openalex_names_ids(supervisor).items()
    }
    
    # Save the data to a CSV file
    print(f"Saving supervisor data to {csv_path}...")
    supervisors_df = pd.DataFrame([
        {"supervisor_name": name, "supervisor_id": openalex_id}
        for name, openalex_id in supervisors_ids.items()
    ])
    makedirs(path.dirname(csv_path), exist_ok=True)  # Ensure the directory exists
    supervisors_df.to_csv(csv_path, index=False)
    
    return supervisors_ids
