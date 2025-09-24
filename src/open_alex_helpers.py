import logging
from pyalex import Authors, Works, config
import pandas as pd
import numpy as np
from os import path, makedirs
from sentence_transformers import util

from src.io_helpers import fetch_supervisors_from_pilot_dataset, remove_illegal_title_characters, ordinal
from src.clean_names_helpers import format_name_to_lastname_firstname, name_sanity_check

class AuthorRelations:
    # Class attribute shared by all instances
    # Can be overwritten at the class level to change the default for all (upcoming) instances in scope
    # Keys: supervisor name 
    # Values: supervisor OpenAlex ID
    supervisors_in_pilot_dataset = dict()
    
    def __init__(self, phd_name, title, year, institution, contributors, model, years_tolerance=0, verbosity='INFO'):
        self.phd_name = phd_name
        self.n_name_search_matches = None # Number of matches for the PhD candidate's name between NARCIS and OpenAlex
        self.title = title # title of the thesis as it appears in Narcis
        self.title_open_alex = None # title of the thesis as it appears in OpenAlex
        self.title_similarities = None # similarities between Narcis title and fuzzily matched OpenAlex titles
        self.max_title_similarity = None # highest similarity between Narcis title and fuzzily matched OpenAlex titles
        self.n_close_matches = None # number of fuzzily matched OpenAlex titles
        self.exact_match = None # True if we have an exact match between Narcis title and OpenAlex title
        self.near_exact_match = None # # True if we have an very good match between Narcis title and OpenAlex title
        self.affiliation_match = None # True if we have a match between Narcis institution and OpenAlex institution
        self.phd_match_score = None # match score for the PhD candidate
        self.thesis_id = None # OpenAlex ID of the thesis
        self.year = year
        self.institution = institution
        self.phd_publications = [] # OpenAlex data for the works of the author with the OpenAlex ID we identified for the PhD candidate
        self.contributors = contributors
        self.years_tolerance = years_tolerance
        self.phd_candidate = None
        self.phd_match_by = None
        self.potential_supervisors = []
        
        # NLP model
        self.model = model
        # Cutoff for considering a title match to be close enough to count as a 'close match'
        self.similarity_cutoff = 0.7
        
        # Define how long before and after the graduation date works can be written to be considered 
        # for the fuzzy title matching to match PhDs
        # The first entry is the years we consider before the graduation date, 
        # the second one is the years after
        # Note: With float("inf") as the first value, we consider all publications that were written 
        # before the graduation
        self.years_offset_phd_matching = [float("inf"), 3]
        
        # Define target years as a property of the object
        self.affiliation_target_years = self.calculate_affiliation_target_years()
        
        # Minimum number of shared publications required for a contributor match 
        self.n_shared_pubs_min = 1
        
        self.verbosity = verbosity.upper()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def calculate_affiliation_target_years(self):
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

    def search_phd_candidate(self):
        """
        Search for the PhD candidate by name and validate by name match, fuzzy match of works and institution.
        Collect all candidates into a list, then decide which best matches criteria.

        Now we also assign a 'match_score' to each candidate:
            match_score = n_close_matches + (50 if exact_match) + (20 if near_exact_match) + (20 if affiliation_match)
        Then we pick the candidate with the highest match_score.

        If in debug mode, print a table representation of the sorted DataFrame.
        """
        self.logger.info(f"Searching for PhD candidate: {self.phd_name}")
        
        # Search for candidates by PhD name
        candidates = Authors().search(self.phd_name).get()
        self.logger.debug(f"Found: {len(candidates)} people who are potential matches.")

        # If no candidates are found, log and return
        if not candidates:
            self.logger.warning("No candidates found with the given PhD name.")
            return None

        # Allocate data frame for works of PhD candidates
        df_works = pd.DataFrame()
        
        # Collect raw and processed info for all candidates
        candidates_info = []
        for candidate in candidates:
                        
            # Some basic sanity checking if the two names could realistically refer to the same person
            if not name_sanity_check(self.phd_name, candidate['display_name']):
                continue
            
            self.logger.debug(f"Evaluating candidate: {candidate['display_name']} (ID: {candidate['id']})")
            affiliation_match = self.check_affiliation(candidate)

            # Retrieve the publications for the current candidate
            df_works_candidate = get_authored_works(author_id=candidate["id"], author_name=candidate["display_name"])
            
            df_works_candidate = compute_and_sort_works_by_title_similarities(
                df_works_candidate, 
                reference_title=self.title, 
                model=self.model
                )
            
            df_works_candidate_in_target_years = get_works_in_target_years(
                df_works_candidate, 
                year=self.year, 
                years_offset=self.years_offset_phd_matching
                )
            
            work_ids_open_alex_in_target_years = df_works_candidate_in_target_years["work_id"].tolist()
            titles_open_alex_in_target_years = df_works_candidate_in_target_years["title"].tolist()
            
            # Calculate the maximum similarity
            if not df_works_candidate_in_target_years.empty: # check if data frame has rows
                title_similarities_in_target_years = df_works_candidate_in_target_years["similarity"].tolist()
                max_similarity = max(title_similarities_in_target_years, default=0.0)
            else:
                title_similarities_in_target_years = []
                max_similarity = 0.0  # No data means similarity is 0.0

            # Quantify degree of match and number of close matches
            
            # We do not require 1.0, because some models like specter are very strict for giving a perfect score.
            # A manual evaluation for specter showed that values of 0.99 and more were always exact matches, with only 
            # non-semantic differences.
            exact_match = (max_similarity >= 0.99)
            near_exact_match = (max_similarity >= 0.9)
            close_matches = [val for val in title_similarities_in_target_years if val >= self.similarity_cutoff]
            n_close_matches = len(close_matches)


            candidates_info.append({
                'candidate': candidate,
                'candidate_name': candidate['display_name'],
                'candidate_id': candidate['id'],
                'ids_open_alex': work_ids_open_alex_in_target_years,
                'titles_open_alex': titles_open_alex_in_target_years,
                'title_similarities': title_similarities_in_target_years,
                'max_similarity': max_similarity,
                'exact_match': exact_match,
                'near_exact_match': near_exact_match,
                'close_matches': close_matches,
                'n_close_matches': n_close_matches,
                'affiliation_match': affiliation_match
            })
            
            # Append the works of the open alex author to the dataframe of all potential works of the phd candidate 
            df_works = pd.concat([df_works, df_works_candidate], ignore_index=True)
        
        # No candidates that passed the name sanity check
        if not candidates_info:
            self.logger.warning("No candidates found that passed the name sanity check with the given PhD name.")
            return None
        
        # Convert to a DataFrame for ranking
        candidates_info_with_scores = pd.DataFrame(candidates_info)

        # Assign 'match_score' using given criteria
        # 1. Number of close matches
        # 2. +50 if we have an exact match
        # 3. +20 if we have a near exact match
        # 4. +20 if we have an affiliation match
        candidates_info_with_scores = candidates_info_with_scores.assign(
            match_score=(
                candidates_info_with_scores['n_close_matches']
                + candidates_info_with_scores['exact_match'].astype(int) * 50
                + candidates_info_with_scores['near_exact_match'].astype(int) * 20
                + candidates_info_with_scores['affiliation_match'].astype(int) * 20
            )
        )

        # Sort by match_score and max_similarity (descending)
        candidates_info_with_scores = candidates_info_with_scores.sort_values(
            by=['match_score', 'max_similarity'],
            ascending=[False, False],
            ignore_index=True
        )


        # If logger is in debug mode, print the ranked table
        if self.logger.isEnabledFor(logging.DEBUG):
            # Select the columns most relevant for debugging the ranking
            columns_to_show = [
                'candidate_name', 'candidate_id', 'match_score',
                'max_similarity', 'n_close_matches', 'exact_match', 'near_exact_match', 'affiliation_match'
            ]
            self.logger.debug(f"Ranked candidates:\n{candidates_info_with_scores[columns_to_show].to_string(index=False)}")

        # Select the row of the best candidate (highest match_score) and convert that to dict
        best_candidate_info = candidates_info_with_scores.iloc[0].to_dict()
        
        # Store the publication of the best candidate in a class variable
        self.phd_publications = df_works.query("author_id == @best_candidate_info['candidate_id']")

        # get the number of name search matches for the candidate name in NARCIS
        self.n_name_search_matches = len(candidates)
        # Assign values to the object for the best match for the candidate
        self.phd_candidate = best_candidate_info['candidate']
        
        # Decide whether we think we confirmed this candidate or not
        criteria_met = best_candidate_info['match_score'] > 0
        
        # For reference, indicate how we arrived at this candidate
        self.phd_match_by = "ranking" if criteria_met else None
        self.title_open_alex = best_candidate_info['titles_open_alex']
        self.title_similarities = best_candidate_info['title_similarities']
        self.max_title_similarity = best_candidate_info['max_similarity']
        self.n_close_matches = best_candidate_info['n_close_matches']
        self.exact_match = best_candidate_info['exact_match']
        self.near_exact_match = best_candidate_info['near_exact_match']
        self.affiliation_match = best_candidate_info['affiliation_match']
        self.phd_match_score = best_candidate_info['match_score']
        
        # Get the thesis id (if present)
        if "title" in self.phd_publications.columns:
            self.thesis_id = (
                self.phd_publications
                .query("title == @self.title_open_alex")
                .first_valid_index()
            )
        else:
            self.thesis_id = None

        self.logger.info(
            f"PhD candidate confirmed by {self.phd_match_by}: {self.phd_candidate['display_name']}"
        )
        self.logger.info(
            f"{len(self.phd_publications)} publications found for that candidate."
        )

        return self.phd_candidate

    def check_affiliation(self, candidate):
        """
        Compare the affiliation of an candidate to `self.institution` in the target years.
        Return True if it matches and False otherwise.
        """
        affiliations = candidate.get('affiliations', [])
        match_found = False

        if self.verbosity == 'DEBUG':
            self.logger.debug(f"Target Institution: '{self.institution}', Target Years: {self.affiliation_target_years}")

        for affiliation in affiliations:
            institution_name = affiliation['institution']['display_name']
            years = affiliation.get('years', [])
            is_match = (self.institution == institution_name) and any(year in self.affiliation_target_years for year in years)
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

    def get_candidate_affiliations(self, candidate, in_target_years=True, must_be_dutch = False):
        """
        Returns a set of institution names that the candidate was affiliated with.

        Parameters:
            candidate (dict): The candidate author object containing affiliation data.
            in_target_years (bool): If True, only include affiliations within the target years.
                                    If False, include all affiliations regardless of year.
            must_be_dutch (bool): If True, check if author every worked a t a Dutch institution
                                    and if not, return empty set.

        Returns:
            set: A set of institution names affiliated with the candidate.
        """
        affiliations = candidate.get('affiliations', [])
        institutions = set()
        
        dutch_institution = False
        
        for affiliation in affiliations:
            institution_name = affiliation['institution']['display_name']
            
            # Verify if the institution name is in the Dutch name translation dictionary
            if affiliation['institution']['country_code'] == 'NL':
                dutch_institution = True
            
            affiliation_years = affiliation.get('years', [])
            
            if not in_target_years or self.affiliation_target_years.intersection(affiliation_years):
                institutions.add(institution_name)
        
        if not must_be_dutch or dutch_institution:
            self.logger.debug(
                f"Found {len(institutions)} affiliation(s) " 
                f"for candidate '{candidate['display_name']}': {institutions} "
                f"{'in target years around graduation.' if in_target_years else 'in any year.'}"
            )
            
            return institutions
        else:
            self.logger.debug(
                f"'{candidate['display_name']}' has not been affiliated with a Dutch institution. Returning empty set."
            )
            
            return set()
            

    def collect_supervision_metadata(self):
        """
        Based on relationships between `self.phd_candidate` and contributors collect metadata that indicates supervision. 
        
        We look up contributors in OpenAlex by name search. Then we check if ANY of the search matches fulfill
        the criteria we define to be a potential supervisor. This implicitly merges all the search results for each contributor name.
        For mote info, see the comment below.
        
        Every contributor yields a results dictionary, even if not confirmed. Unmatched contributors get placeholder values.
        The following metadata will be collected per contributor:
        'contributor_name_narcis': original name used from NARCIS -> str
        'contributor_rank': Rank of contributor based on the order they are mentioned in the dataset -> int
        'supervisor': List of matched OpenAlex candidate records -> list
        'supervisor_confirmed': True if confirmation criteria met -> bool
        'same_grad_inst': True if PhD and any candidate share institution at graduation -> bool
        'n_shared_inst_grad': Number of institutions PhD candidate and contributors share at graduation -> int
        'is_sup_in_pilot_dataset': True if any candidate is in pilot dataset -> bool
        'sup_match_by': Description of matching criterion -> str
        'n_shared_pubs': Total number of shared publication DOIs -> int
        'shared_pubs': List of shared publication DOIs -> list
        'is_thesis_coauthor': True if any candidate coauthored the thesis -> bool
        """
        
        # search for contributor if we managed to find and confirm the PhD
        if not self.phd_candidate or not self.phd_match_by:
            self.logger.warning("PhD candidate not confirmed. Cannot find potential supervisors.")
            return []

        # Get PhD candidate's affiliations at graduation
        phd_affiliations_at_graduation = self.get_candidate_affiliations(
            self.phd_candidate, in_target_years=True
        )
        if not phd_affiliations_at_graduation:
            self.logger.warning(
                "PhD candidate has no affiliations in target years. Cannot find potential supervisors."
            )
            return []

        self.logger.debug(
            f"Target Institutions: {phd_affiliations_at_graduation}, Target Years: {self.affiliation_target_years}"
        )
        self.logger.info("Searching for potential supervisors among contributors.")

        self.potential_supervisors = []
        phd_dois = self.phd_publications["doi"].tolist()

        for idx, contributor_name in enumerate(self.contributors):
            contributor_rank = idx + 1
            self.logger.debug(f"Processing contributor #{contributor_rank}: {contributor_name}")

            # Search for contributors in OpenAlex
            openalex_candidates = Authors().search(contributor_name).get()
            self.logger.debug(
                f"Found {len(openalex_candidates)} OpenAlex candidates for '{contributor_name}'."
            )
            
            # Allocate dict for aggregated supervisor data
            supervisor_data = {
                'contributor_name_narcis': contributor_name,
                'name_matches_open_alex': [],
                'contributor_rank': contributor_rank,
                'supervisor': openalex_candidates,
                'supervisor_confirmed': False,
                'same_grad_inst': False,
                'n_shared_inst_grad': 0,
                'is_sup_in_pilot_dataset': False,
                'sup_match_by': '',
                'n_shared_pubs': 0,
                'shared_pubs': [],
                'is_thesis_coauthor': False
            }

            if not openalex_candidates:
                self.logger.debug(
                    f"No OpenAlex matches for '{contributor_name}'. Adding placeholder entry."
                )
                self.potential_supervisors.append(supervisor_data)
                continue

            # Identify candidates with either shared institution or shared publications
            
            # Create placeholder data that we are also using if we don't confirm the supervisor
            name_matches_open_alex = [] # Name match with OpenAlex
            shared_pub_union = set()
            all_shared_affils = set()
            coauthorship_flag = False
            same_grad_inst_flag = False
            pilot_flag = False
            thesis_coauthor_flag = False

            # Open Alex has a lot partial duplicates of authors, especially for ones that are
            # later in their career.
            # We thus decided to implicitly merge all of the matches, i.e. candidates that we find with
            # based on the Open Alex name search and that have te correct affiliation in the target years.
            # This means that we set the boolean flags to True if they apply to (at least) ONE of the
            # matched potential contributors and that we collect the shared publication between the
            # PhD candidate and ALL of the matched potential contributors. 
            for candidate in openalex_candidates:
                
                # Some basic sanity checking if the two names could realistically refer to the same person
                if not name_sanity_check(contributor_name, candidate['display_name']):
                    continue
                
                name_matches_open_alex.append(candidate['display_name'])

                # Affiliations
                cand_affils = self.get_candidate_affiliations(
                    candidate, in_target_years=True, must_be_dutch=True
                )
                shared_affils = phd_affiliations_at_graduation.intersection(cand_affils)

                all_shared_affils.update(shared_affils)

                # Publications
                works = get_authored_works(
                    author_id=candidate['id'], author_name=candidate['display_name']
                )
                contrib_dois = set(works['doi'].tolist())
                shared_pubs = set(phd_dois).intersection(contrib_dois)
                shared_pub_union.update(shared_pubs)

                if len(shared_pub_union) >= self.n_shared_pubs_min:
                    coauthorship_flag = True

                if self.thesis_id in works['work_id'].tolist():
                    thesis_coauthor_flag = True

                if self.institution in shared_affils:
                    same_grad_inst_flag = True

                if candidate['id'] in self.__class__.supervisors_in_pilot_dataset.values():
                    pilot_flag = True
                
                self.logger.debug(
                    f"Processing name match '{candidate['display_name']}' for NARCIS name '{contributor_name}': "
                    f"{len(shared_pubs)} shared publications, "
                    f"{'thesis coauthor' if thesis_coauthor_flag else 'not thesis coauthor'}, "
                    f"{'same graduation institution' if same_grad_inst_flag else 'not same graduation institution'}, and "
                    f"{'in pilot dataset' if pilot_flag else 'not in pilot dataset'}"
                )

            # Check match criteria
            criteria_met = coauthorship_flag # We require at least one shared publication.
            sup_match_by = f"Name match and {self.n_shared_pubs_min}+ shared publications."
            
            # criteria_met = same_grad_inst_flag # We require at least one shared affiliation in the target years around graduation
            # sup_match_by = "Shared affiliation at graduation."
            
            # Fill aggregated values
            supervisor_data.update({
                "name_matches_open_alex":     name_matches_open_alex,
                "supervisor":                 openalex_candidates,
                "n_shared_inst_grad":         len(all_shared_affils),
                "same_grad_inst":             same_grad_inst_flag,
                "is_sup_in_pilot_dataset":    pilot_flag,
                "n_shared_pubs":              len(shared_pub_union),
                "shared_pubs":                list(shared_pub_union),
                "is_thesis_coauthor":         thesis_coauthor_flag,
                "supervisor_confirmed":       criteria_met,
                "sup_match_by": (
                    sup_match_by if criteria_met else ''
                ),
            })
            
            # Logging match or not
            if criteria_met:                
                self.logger.info(
                    f"Contributor '{contributor_name}' matched by {supervisor_data['sup_match_by']}"
                )
            else:
                self.logger.debug(
                    f"Matching criteria not met for '{contributor_name}'."
                )
                
            # Append the data before moving to the next supervisor listed in NARCIS
            self.potential_supervisors.append(supervisor_data)

        self.logger.info(
            f"Processed {len(self.contributors)} contributors; "
            f"{sum(1 for s in self.potential_supervisors if s['supervisor_confirmed'])} confirmed supervisors."
        )
        return self.potential_supervisors

    def get_results(self):
        """
        Return a DataFrame with the results of the extraction.

        If no PhD candidate was found in OpenAlex, return a single-row DataFrame
        with only 'phd_name' filled and all other columns as None.

        If a PhD candidate was found but no supervisors were confirmed,
        also return a single-row DataFrame with 'phd_name', 'phd_id', 'phd_match_by' filled
        and all supervisor-related columns as NaN.
        """
        
        # The columns our DataFrame should have
        columns = [
            'phd_name', 
            'phd_id', 
            'n_name_search_matches',
            'year', 
            'title', 
            'title_open_alex', 
            'title_similarities',
            'max_title_similarity',
            'n_close_matches',
            'exact_match',
            'near_exact_match',
            'affiliation_match',
            'phd_match_score',
            'phd_match_by',
            'contributor_name_narcis',
            'name_matches_open_alex',
            'contributor_confirmed',
            'contributor_name', 
            'contributor_id',
            'sup_match_by',
            'contributor_rank', 
            'same_grad_inst', 
            'n_shared_inst_grad', 
            'is_sup_in_pilot_dataset', 
            'n_shared_pubs', 
            'shared_pubs', 
            'is_thesis_coauthor'
        ]

        
        if not self.phd_candidate:
            self.logger.warning("PhD candidate was not found in Open Alex so we can't look for contributors either")
            # Create a single row with the data we have and the others as None
            result_row = {col: None for col in columns}
            result_row['phd_name'] = self.phd_name
            result_row['n_name_search_matches'] = 0
            result_row['year'] = self.year
            result_row['title'] = self.title
            return pd.DataFrame([result_row], columns=columns)

        # If we reach this, we have a confirmed the PhD candidate
        phd_id = self.phd_candidate['id']
        phd_name = self.phd_candidate['display_name']
        title_open_alex = self.title_open_alex if self.title_open_alex else None # convert empty list to None
        title_similarities = self.title_similarities if self.title_similarities else None # convert empty list to None
        max_title_similarity = self.max_title_similarity if self.max_title_similarity else None
        
        # Create a list of dictionaries for each supervisor
        # Each supervisor is represented by one row in the final dataset
        results_list = []
        for supervisor_data in self.potential_supervisors:
            supervisor = supervisor_data['supervisor']
            # with the implicit merging of all potential contributors, the contributor names and ids become lists
            contributor_name_narcis = supervisor_data["contributor_name_narcis"]
            name_matches_open_alex = supervisor_data["name_matches_open_alex"]
            contributor_confirmed = supervisor_data["supervisor_confirmed"]
            contributor_name = [supervisor_nested['display_name'] for supervisor_nested in supervisor]
            contributor_id = [supervisor_nested['id'] for supervisor_nested in supervisor]
            sup_match_by = supervisor_data['sup_match_by']
            contributor_rank = supervisor_data['contributor_rank']
            same_grad_inst = supervisor_data['same_grad_inst']
            n_shared_inst_grad = supervisor_data['n_shared_inst_grad']
            is_sup_in_pilot_dataset = supervisor_data['is_sup_in_pilot_dataset']
            shared_pubs = supervisor_data['shared_pubs']
            is_thesis_coauthor = supervisor_data['is_thesis_coauthor']

            result_row = {
                'phd_name': phd_name,
                'phd_id': phd_id,
                'n_name_search_matches': self.n_name_search_matches,
                'year': self.year,
                'title': self.title,
                'title_open_alex': title_open_alex,
                'title_similarities': title_similarities,
                'max_title_similarity': max_title_similarity,
                'n_close_matches': self.n_close_matches,
                'exact_match': self.exact_match,
                'near_exact_match': self.near_exact_match,
                'affiliation_match': self.affiliation_match,
                'phd_match_score': self.phd_match_score,
                'phd_match_by': self.phd_match_by,
                'contributor_name_narcis': contributor_name_narcis,
                'name_matches_open_alex': name_matches_open_alex,
                'contributor_confirmed': contributor_confirmed,
                'contributor_name': contributor_name,
                'contributor_id': contributor_id,
                'sup_match_by': sup_match_by,
                'contributor_rank': contributor_rank,
                'same_grad_inst': same_grad_inst,
                'n_shared_inst_grad': n_shared_inst_grad,
                'is_sup_in_pilot_dataset': is_sup_in_pilot_dataset,
                'n_shared_pubs': len(shared_pubs),
                'shared_pubs': shared_pubs,
                'is_thesis_coauthor': is_thesis_coauthor
            }
            results_list.append(result_row)
        
        if not results_list:
            self.logger.warning("PhD candidate confirmed, but no supervisors found.")
            # Create a single row with the data we have and the others as None
            result_row = {col: None for col in columns}
            result_row['phd_name'] = phd_name
            result_row['phd_id'] = phd_id
            result_row['n_name_search_matches'] = self.n_name_search_matches
            result_row['year'] = self.year
            result_row['title'] = self.title
            result_row['title_open_alex'] = self.title_open_alex if self.title_open_alex else None # convert empty list to None
            result_row['title_similarities'] = self.title_similarities if self.title_similarities else None
            result_row['max_title_similarity'] = self.max_title_similarity if self.max_title_similarity else None
            result_row['n_close_matches'] = self.n_close_matches
            result_row['exact_match'] = self.exact_match
            result_row['near_exact_match'] = self.near_exact_match
            result_row['affiliation_match'] = self.affiliation_match
            result_row['phd_match_score'] = self.phd_match_score
            result_row['phd_match_by'] = self.phd_match_by
            # The supervisor-related columns remain None
            results_df = pd.DataFrame([result_row], columns=columns)
        else:
            results_df = pd.DataFrame(results_list, columns=columns)

        return results_df


def get_authored_works(author_id: str, author_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing the works authored by the candidate.
    """ 
    
    properties_to_select = ["id", "title", "doi", "publication_year", "type"]
    
    # Do the API call
    works = pd.DataFrame(
        Works()
        .filter(author={"id": author_id})
        .select(properties_to_select)
        .get(),
        columns=properties_to_select # make sure the dataframe has these columns, even when it's empty
        )
    
    # Make it clear that the id we got here is the work id
    works = works.rename(columns={'id': 'work_id'})

    # Add the authorship of the author we're looking at
    works.insert(0, "author_id", author_id)
    works.insert(1, "author_name", author_name)
    
    return works

def get_works_in_target_years(works: pd.DataFrame, year: int, years_offset: list[int]) -> pd.DataFrame:
    """
    Filter out the works in the target years from a DataFrame containing works.

    Parameters:
    - works (pd.DataFrame): Dataframe containing works.
    - year (int): The reference year.
    - years_offset (list): Allowed year offsets.

    Returns:
    - pd.DataFrame: works, but with all the publications outside the target years removed
    """
        
    if not works.empty:
        # Minimum and maximum years we want to consider publications from
        # These variables might appear unnused, but they are used in query below
        min_year = year - years_offset[0]
        max_year = year + years_offset[1]
        
        # Filter out publications outside of the range that we want to consider
        return works.query("@min_year <= publication_year <= @max_year")
        
    else:
        return works
    
    
def compute_and_sort_works_by_title_similarities(works: pd.DataFrame, reference_title: str, model) -> pd.DataFrame:
    """
    Computes the similarity between each title in the given Series and the reference title.

    Parameters:
    - works (pd.DataFrame): Dataframe containing works.
    - reference_title (str): The reference title to compare the work titles against.
    - model: The similarity model.

    Returns:
    - pd.DataFrame: works, but with a new column similarity and with the rows sorted descending by similarity
    """
    
    reference_title_norm = (
            # Remove illegal characters from the title and lowercase to make search() robust 
            # (most importantly remove pipe characters "|", which search() interprets as OR)
            # Lowercase the title mostly to allow a more reasonable similarity calculation later.
            # Similarity is very sensitive to capitalization, but since that is not very consistent between
            # OpenAlex and Narcis, we get rid of it now.
            
            # if we don't get a string back, replace with an empty string so .similarity() doesn't error out
            remove_illegal_title_characters(reference_title).lower() 
            if isinstance(reference_title, str) 
            else "" 
    )
    
    # Encode the reference title if valid.
    emb1 = (
        model.encode(reference_title_norm, convert_to_tensor=True, show_progress_bar=False)
        if reference_title_norm
        else None
    )
    
    # Process each title in 'works["title"]', skipping missing or empty strings.
    # Skip, if reference title was not valid or if the works df has no rows
    if emb1 is None and not works.empty:
        title_similarities = [np.nan] * len(works)
    else:
        title_similarities = []
        for title in works["title"]:
            if isinstance(title, str) and title.strip():
                # same processing as for reference title
                processed_title = remove_illegal_title_characters(title).lower()
                emb2 = model.encode(processed_title, convert_to_tensor=True, show_progress_bar=False)
                
                # compute cosine similarity
                similarity = util.cos_sim(emb1, emb2).item() if emb2 is not None else np.nan
            else:
                similarity = np.nan
            title_similarities.append(similarity)
        
    # assign to column of np.nan similarity
    works["similarity"] = title_similarities if title_similarities else [np.nan] * len(works)

    works = works.sort_values("similarity", ascending=False)
    
    return works


def find_phd_and_supervisors_in_row(row, model):
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
        model=model,
        years_tolerance=-4, # cf. issue #19
        verbosity='DEBUG'  # Set to 'NONE' for production
    )
    
    # Search for the PhD candidate
    author_relations.search_phd_candidate()
    
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
