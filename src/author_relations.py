import logging
from pyalex import Authors, Works
import pandas as pd

class AuthorRelations:
    def __init__(self, phd_name, title, year, institution, contributors, years_tolerance=0, verbosity='INFO'):
        self.phd_name = phd_name
        self.title = title
        self.year = year
        self.institution = institution
        self.contributors = contributors
        self.years_tolerance = years_tolerance  # Changed from 'tolerance' to 'years_tolerance'
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
            return list(range(self.year, self.year + self.years_tolerance + 1))
        else:  # years_tolerance < 0
            return list(range(self.year + self.years_tolerance, self.year + 1))
        
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
        Check if the candidate has the correct affiliation in the target years.
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

    def get_candidate_affiliations(self, candidate):
        """
        Returns a set of institution names that the candidate was affiliated with in the target years.
        """
        affiliations = candidate.get('affiliations', [])
        institutions = set()
        for affiliation in affiliations:
            institution_name = affiliation['institution']['display_name']
            affiliation_years = affiliation.get('years', [])
            if any(year in self.target_years for year in affiliation_years):
                institutions.add(institution_name)
        return institutions

    def find_potential_supervisors(self):
        """
        Find potential supervisors among the contributors based on shared affiliations with the PhD candidate
        in the target years.
        """
        if not self.phd_candidate:
            self.logger.warning("PhD candidate not confirmed. Cannot find potential supervisors.")
            return []

        # Get PhD candidate's affiliations in target years
        phd_affiliations = self.get_candidate_affiliations(self.phd_candidate)
        if not phd_affiliations:
            self.logger.warning("PhD candidate has no affiliations in target years. Cannot find potential supervisors.")
            return []

        # Log the target institutions (affiliations of the PhD candidate in the target years)
        self.logger.debug(f"Target Institutions: {phd_affiliations}, Target Years: {self.target_years}")
        self.logger.info("Searching for potential supervisors among contributors.")

        for contributor_name in self.contributors:
            self.logger.debug(f"Searching for contributor: {contributor_name}")
            # Search for contributors in OpenAlex
            candidates = Authors().search(contributor_name).get()
            self.logger.debug(f"Found: {len(candidates)} candidates for contributor '{contributor_name}'.")

            # If no candidates are found, log and continue to next contributor
            if not candidates:
                self.logger.debug(f"No candidates found for contributor: {contributor_name}")
                continue

            supervisor_found = False
            for candidate in candidates:
                # Get supervisor's affiliations in target years
                supervisor_affiliations = self.get_candidate_affiliations(candidate)

                # Check for shared affiliations
                shared_affiliations = phd_affiliations.intersection(supervisor_affiliations)

                # Logging per institution we are checking
                for institution in supervisor_affiliations:
                    is_match = institution in phd_affiliations
                    self.logger.debug(
                        f"Checking affiliation: Potential Supervisor '{candidate['display_name']}' Institution '{institution}' - "
                        f"Match Found: {'Yes' if is_match else 'No'}"
                    )

                if shared_affiliations:
                    self.potential_supervisors.append(candidate)
                    self.logger.info(f"Potential supervisor found: {candidate['display_name']} with shared institutions {shared_affiliations}")
                    supervisor_found = True
                    break  # Assuming the first match suffices

            if not supervisor_found:
                self.logger.debug(f"No shared affiliations found for contributor: {contributor_name}")

        # Log the total number of contributors with matches
        self.logger.info(
            f"Total contributors with matches: {len(self.potential_supervisors)} out of {len(self.contributors)}"
        )

        if not self.potential_supervisors:
            self.logger.warning("No potential supervisors found.")
        return self.potential_supervisors

    def get_results(self):
        """
        Return the OpenAlex ID pairs where matches are found.
        """
        if not self.phd_candidate:
            self.logger.warning("No results to return; PhD candidate was not found.")
            return None
        phd_id = self.phd_candidate['id']
        supervisor_ids = [supervisor['id'] for supervisor in self.potential_supervisors]
        self.logger.info(f"Returning results: PhD ID {phd_id}, Supervisor IDs {supervisor_ids}")
        return {'phd_id': phd_id, 'supervisor_ids': supervisor_ids}
    
    
# Define a function to process each row
def find_phd_and_supervisors_in_row(row):
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
    author_relations.find_potential_supervisors()
    
    # Get the OpenAlex ID pairs
    results = author_relations.get_results()
    
    # Return the results along with the original row index or any additional data you need
    return {
        'phd_name': phd_name,
        'phd_id': results.get('phd_id') if results else None,
        'supervisor_ids': results.get('supervisor_ids') if results else None
    }