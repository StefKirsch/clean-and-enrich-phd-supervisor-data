import logging
from pyalex import Authors, Works, config
import pandas as pd
from os import path, makedirs
import time
from requests.exceptions import ConnectionError, ReadTimeout
from sentence_transformers import util

from src.io_helpers import fetch_supervisors_from_pilot_dataset, remove_illegal_title_characters, ordinal
from src.clean_names_helpers import format_name_to_lastname_firstname

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
        self.max_title_similarity = None # highest similarity between Narcis title and fuzzily matched OpenAlex titles
        self.n_close_matches = None # number of fuzzily matched OpenAlex titles
        self.exact_match = None # True if we have an exact match between Narcis title and OpenAlex title
        self.near_exact_match = None # # True if we have an very good match between Narcis title and OpenAlex title
        self.affiliation_match = None # True if we have a match between Narcis institution and OpenAlex institution
        self.phd_match_score = None # match score for the PhD candidate
        self.thesis_id = None # OpenAlex ID of the thesis
        self.year = year
        self.institution = institution
        self.phd_publications = []
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
        candidates = AuthorsWithRetry().search(self.phd_name).get()
        self.logger.debug(f"Found: {len(candidates)} people who are potential matches.")

        # If no candidates are found, log and return
        if not candidates:
            self.logger.warning("No candidates found with the given PhD name.")
            return None

        # Collect raw and processed info for all candidates
        candidates_info = []
        for candidate in candidates:
            self.logger.debug(f"Evaluating candidate: {candidate['display_name']} (ID: {candidate['id']})")
            affiliation_match = self.check_affiliation(candidate)

            # This returns lists for the OpenAlex IDs, the titles, and the similarities
            # sorted in descending order by similarity.
            ids_open_alex, titles_open_alex, title_similarities = self.check_authored_work(candidate)
            
            # TODO Potential Optimization
            # instead of just getting the dissertation and similarity here, we can also get all the
            # works of the candidate right away here as a works object and convert it to a dataframe
            # we can then perform all checks on that dataframe. If we manage to confirm the candidate as the PhD,
            # we break out of the loop, but the df sticks around. So we can then do all further checks and
            # operations on the dataframe we got from the confirmed candidate
            # We can then completely get rid of the `check_authored_work` method.
            # Maybe we can add a method for the checking logic though.
        
            max_similarity = title_similarities[0] if len(title_similarities) > 0 else 0.0

            # Quantify degree of match and number of close matches
            
            # We do dot require 1.0, because some models like specter are very strict for giving a perfect score.
            # A manual evaluation for specter showed that values of 0.99 and more were always exact matches, with only 
            # non-semantic differences.
            exact_match = (max_similarity >= 0.99)
            near_exact_match = (max_similarity >= 0.9)
            close_matches = [val for val in title_similarities if val >= self.similarity_cutoff]
            n_close_matches = len(close_matches)

            candidates_info.append({
                'candidate': candidate,
                'candidate_name': candidate['display_name'],
                'candidate_id': candidate['id'],
                'ids_open_alex': ids_open_alex,
                'titles_open_alex': titles_open_alex,
                'title_similarities': title_similarities,
                'max_similarity': max_similarity,
                'exact_match': exact_match,
                'near_exact_match': near_exact_match,
                'close_matches': close_matches,
                'n_close_matches': n_close_matches,
                'affiliation_match': affiliation_match
            })

        # Convert to a DataFrame for ranking
        df = pd.DataFrame(candidates_info)

        # Assign 'match_score' using given criteria
        # 1. Number of close matches
        # 2. +50 if we have an exact match
        # 3. +20 if we have a near exact match
        # 4. +20 if we have an affiliation match
        df = df.assign(
            match_score=(
                df['n_close_matches']
                + df['exact_match'].astype(int) * 50
                + df['near_exact_match'].astype(int) * 20
                + df['affiliation_match'].astype(int) * 20
            )
        )

        # Sort by match_score descending
        df = df.sort_values('match_score', ascending=False, ignore_index=True)

        # If logger is in debug mode, print the ranked table
        if self.logger.isEnabledFor(logging.DEBUG):
            # Select the columns most relevant for debugging the ranking
            columns_to_show = [
                'candidate_name', 'candidate_id', 'match_score',
                'max_similarity', 'n_close_matches', 'exact_match', 'near_exact_match', 'affiliation_match'
            ]
            self.logger.debug(f"Ranked candidates:\n{df[columns_to_show].to_string(index=False)}")

        # Optional check: If all scores are zero or df is empty, we might want to stop here
        if df.empty or df.loc[0, 'match_score'] == 0:
            self.logger.warning("No candidates meet the given ranking criteria.")
            return None

        # Select the best candidate (highest match_score)
        best_candidate_info = df.iloc[0].to_dict()

        # get the number of name search matches for the candidate name in NARCIS
        self.n_name_search_matches = len(candidates)
        # Assign values to the object for the best match for the candidate
        self.phd_candidate = best_candidate_info['candidate']
        # For reference, indicate how we arrived at this candidate
        self.phd_match_by = "ranking"
        self.title_open_alex = best_candidate_info['titles_open_alex']
        self.max_title_similarity = best_candidate_info['max_similarity']
        self.n_close_matches = best_candidate_info['n_close_matches']
        self.exact_match = best_candidate_info['exact_match']
        self.near_exact_match = best_candidate_info['near_exact_match']
        self.affiliation_match = best_candidate_info['affiliation_match']
        self.phd_match_score = best_candidate_info['match_score']

        # Retrieve the publications for the best candidate
        # TODO with the above optimization, we can move this up
        self.phd_publications = pd.DataFrame(
            WorksWithRetry()
            .filter(author={"id": self.phd_candidate['id']})
            .select(["id", "title", "doi", "type"]).get()
        )
        
        # Get the thesis id (if present)
        self.thesis_id = (
            self.phd_publications
            .query("title == @self.title_open_alex")
            .first_valid_index()
        )

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
            
            if not in_target_years or self.affiliation_target_years.intersection(affiliation_years):
                institutions.add(institution_name)
        return institutions
    
    def check_authored_work(self, candidate):
        """
        Check if the candidate has authored the specified title.
        """

        # Pre-process the title so that it works with the search parameter and with nlp() and .similarity()
        title_search_str = (
            # Remove illegal characters from the title and lowercase to make search() robust 
            # (most importantly remove pipe characters "|", which search() interprets as OR)
            # Lowercase the title mostly to allow a more reasonable similarity calculation later.
            # Similarity is very sensitive to capitalization, but since that is not very consistent between
            # OpenAlex and Narcis, we get rid of it now.
            remove_illegal_title_characters(self.title).lower()
            if isinstance(self.title, str) 
            # if we don't get a string back, replace with an empty string so .similarity() doesn't error out
            else ""
            )
                
        # Get the title of the PhD candidate's dissertation.
        # WorksWithRetry returns a list, so if there are several matching works, we get all of them 
        # We then convert it into a dataframe to handle in more easily
        works_by_candidate = pd.DataFrame(
            WorksWithRetry()
                # Search for the title
                # We don't do this right now so that we are not dependent on the search matching of
                # Open Alex
                #.search(title_search_str)
                .filter(author={"id": candidate['id']})
                # Require work to be listed as a dissertation
                # This is commented out right now, the main reason being that many dissertations aren't
                # listed as such in OpenAlex 
                #.filter(type="dissertation")
                .select(["id", "title", "publication_year"])
                .get()  # get returns a dict with the selected properties as key-value pairs.
        )
        
        # Return empty lists if no works were found
        if len(works_by_candidate) == 0:
            return [], [], []
        
        # Minimum and maximum years we want to consider publications from
        # These variables might appear unnused, but they are used in query below
        min_year = self.year - self.years_offset_phd_matching[0]
        max_year = self.year + self.years_offset_phd_matching[1]
        
        # Filter out publications outside of the range that we want to consider
        works_by_candidate = works_by_candidate.query("@min_year <= publication_year <= @max_year")
        
        # Convert the list of dicts to a list of values, extracting the title(s)
        ids_open_alex = works_by_candidate['id'].tolist()
        titles_open_alex = works_by_candidate['title'].tolist()
        
        # Get semantic similarity between Narcis title and OpenAlex title with NLP model
         
        # doc1.similarity(doc2) is sensitive to erroring out, so we make sure that we start out with valid strings
        titles_str_open_alex = [
            # We do the same processing with the OpenAlex title as we do with the Narcis title
            remove_illegal_title_characters(title).lower()
            for title in titles_open_alex
            if isinstance(title, str)
            ]
        
        # Return empty lists if no works were found or only works with no title after character removal
        # Note: This effectively rejects Works with no title, which seems reasonable 
        if not titles_str_open_alex:
            return [], [], []

        if title_search_str and titles_str_open_alex:
            # Encode string document
            emb1 = self.model.encode(title_search_str, convert_to_tensor=True, show_progress_bar=False)

            title_similarities = []
            
            for title in titles_str_open_alex:
                emb2 = self.model.encode(title, convert_to_tensor=True, show_progress_bar=False)
                
                # if we have a proper encoding for both documents, we calculate similarity
                if emb1 is not None and emb2 is not None:
                    similarity = util.cos_sim(emb1, emb2)
                # otherwise we output 0
                else:
                    similarity = 0.0
                    
                title_similarities.append(similarity.item())
        else:
            title_similarities = []  # No valid data to compare

        # Sort titles by similarity
        combined = list(zip(ids_open_alex, titles_open_alex, title_similarities))
        # Sort by the score (3rd element)
        combined.sort(key=lambda x: x[2], reverse=True)
        # Unzip back
        sorted_ids, sorted_titles, sorted_similarities = zip(*combined)
        
        self.logger.debug(
            f"Finding publications by '{candidate['id']}' - Found {len(sorted_titles)} works"
            f"with the following similarity scores to their dissertation {self.title}: {sorted_similarities}. {sorted_titles[0]} is the closest match."
        )
        
        # Return various metrics about the candidate works, sorted by similarity
        return sorted_ids, sorted_titles, sorted_similarities
            

    def collect_supervision_metadata(self):
        """
        Based on relationships between `self.phd_candidate` and contributors collect metadata that indicates supervision. 
        
        We look up contributors in OpenAlex. We confirm that we matched the contributor name with an author in OpenAlex
        by requiring at least *one* shared affiliation in the time window (target years) of the publication of the thesis.  
        
        Every confirmed contributor is added to the final dataset, the only difference for them will be the metadata that we collect here
        
        The following metadata will be collected per confirmed contributor:
        'contributor_rank': Rank of contributor based on the order they are mentioned in the dataset -> int
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
        self.logger.debug(f"Target Institutions: {phd_affiliations_at_graduation}, Target Years: {self.affiliation_target_years}")
        self.logger.info("Searching for potential supervisors among contributors.")
        
        # Initialize list to store supervisor data
        self.potential_supervisors = []

        for idx, contributor_name in enumerate(self.contributors):
            self.logger.debug(f"Searching for contributor: {contributor_name}")

            # set flag for candidate being the first contributor in the underlying dataset
            contributor_rank = idx + 1
            
            # Search for contributors in OpenAlex
            openalex_candidates = AuthorsWithRetry().search(contributor_name).get()
            self.logger.debug(f"Found: {len(openalex_candidates)} candidates for contributor '{contributor_name}'.")
            
            # If no candidates are found continue to next contributor
            if not openalex_candidates:
                self.logger.debug(f"No candidates found for contributor: {contributor_name}. Moving to next contributor.")
                continue
            
            # Prepare contributor_found_in_openalex flag
            contributor_found_in_openalex = False
            
            # Use a list comprehension to collect all candidates with a shared affiliation
            matching_candidates = [
                candidate for candidate in openalex_candidates
                if phd_affiliations_at_graduation.intersection(
                    self.get_candidate_affiliations(candidate, in_target_years=True)
                )
            ]
            # Log the number of affiliation matches among candidates
            self.logger.debug(
                f"Found {len(matching_candidates)} affiliation match(es) among {len(openalex_candidates)} search matches for name '{contributor_name}'"
            )

            # Open Alex has a lot partial duplicates of authors, especially for ones that are
            # later in their career.
            # We thus decided to implicitly merge all of the matches, i.e. candidates that we find with
            # based on the Open Alex name search and that have te correct affiliation in the target years.
            # This means that we set the boolean flags to True if they apply to (at least) ONE of the
            # matched potential contributors and that we collect the shared publication between the
            # PhD candidate and ALL of the matched potential contributors. 
            if matching_candidates:
                # Initialize aggregated variables for merging candidate info
                all_shared_affiliations = set()
                same_grad_inst = False
                is_sup_in_pilot_dataset = False
                shared_publications_union = set()
                is_thesis_coauthor = False

                # Get PhD candidate publication DOIs once
                phd_dois = self.phd_publications["doi"].tolist()

                # Loop through each matching candidate
                for id_cand, candidate in enumerate(matching_candidates):
                    
                    self.logger.debug(
                        f"Evaluating and merging data for the {ordinal(id_cand+1)} of {len(matching_candidates)} potential contributors: {candidate['display_name']} ({candidate['id']})"
                    )
                    
                    candidate_affiliations = self.get_candidate_affiliations(candidate, in_target_years=True)
                    candidate_shared_affils = phd_affiliations_at_graduation.intersection(candidate_affiliations)
                    all_shared_affiliations.update(candidate_shared_affils)
                    
                    # Check if candidate meets same_grad_inst criterion
                    if self.institution in candidate_shared_affils:
                        same_grad_inst = True
                    
                    # Check if candidate is in the pilot dataset
                    if candidate['id'] in self.__class__.supervisors_in_pilot_dataset.values():
                        is_sup_in_pilot_dataset = True
                    
                    # Query publications for candidate contributor
                    contrib_publications = WorksWithRetry() \
                        .filter(author={"id": candidate['id']}) \
                        .select(["id", "doi"]) \
                        .get()
                    contrib_dois = [pub["doi"] for pub in contrib_publications]
                    # Merge shared publication DOIs across candidates and add to set
                    # of earlier found shared publications
                    shared_publications_union.update(set(phd_dois).intersection(contrib_dois))
                    
                    # Check if candidate is a thesis coauthor
                    if self.thesis_id in [pub["id"] for pub in contrib_publications]:
                        is_thesis_coauthor = True

                if len(shared_publications_union) < self.n_shared_pubs_min:
                    self.logger.debug("No shared publications found between PhD candidate and potential contributors. Moving to next contributor.")
                    continue
                
                n_shared_inst_grad = len(all_shared_affiliations)
                shared_dois = list(shared_publications_union)

                # Build the merged supervisor data
                supervisor_data = {
                    'supervisor': matching_candidates,  # list of all matching candidates
                    'contributor_rank': contributor_rank,
                    'same_grad_inst': same_grad_inst,  # True if at least one candidate has the same graduation institution
                    'n_shared_inst_grad': n_shared_inst_grad,  # total count of shared institutions across candidates
                    'is_sup_in_pilot_dataset': is_sup_in_pilot_dataset,  # True if at least one candidate is in the pilot dataset
                    'sup_match_by': f"Name match, shared institution at graduation and >={self.n_shared_pubs_min} shared publications.",
                    'n_shared_pubs': len(shared_dois),  # total number of shared DOIs
                    'shared_pubs': shared_dois,  # list of shared publication DOIs
                    'is_thesis_coauthor': is_thesis_coauthor  # True if at least one candidate coauthored the thesis
                }
                self.potential_supervisors.append(supervisor_data)
                self.logger.info(
                    f"Merge of {len(matching_candidates)} potential contributors yielded {n_shared_inst_grad} shared institution(s) at graduation and {len(shared_dois)} shared publication(s)."
                )
                contributor_found_in_openalex = True


            if not contributor_found_in_openalex:
                self.logger.debug(f"No shared affiliations found for contributor: {contributor_name}. Moving to next contributor.")

        # Log the total number of contributors with matches
        self.logger.info(
            f"Total contributors with shared affiliations: {len(self.potential_supervisors)} out of {len(self.contributors)}"
        )

        if not self.potential_supervisors:
            self.logger.warning("No potential supervisors found.")
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
            'max_title_similarity',
            'n_close_matches',
            'exact_match',
            'near_exact_match',
            'affiliation_match',
            'phd_match_score',
            'phd_match_by',
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
        max_title_similarity = self.max_title_similarity if self.max_title_similarity else None
        
        # Create a list of dictionaries for each supervisor
        # Each supervisor is represented by one row in the final dataset
        results_list = []
        for supervisor_data in self.potential_supervisors:
            supervisor = supervisor_data['supervisor']
            # with the implicit merging of all potential contributors, the contributor names and ids are become lists
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
                'max_title_similarity': max_title_similarity,
                'n_close_matches': self.n_close_matches,
                'exact_match': self.exact_match,
                'near_exact_match': self.near_exact_match,
                'affiliation_match': self.affiliation_match,
                'phd_match_score': self.phd_match_score,
                'phd_match_by': self.phd_match_by,
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


# Wrapper classes for Authors() and Works() with backoff and retry logic
class AuthorsWithRetry(Authors):
    """
    Wrapper around pyalex.Authors that retries on ConnectionError/ReadTimeout.
    Usage example:
        candidates = AuthorsWithRetry().search("John Doe").get()
    """
    
    # Force the endpoint to remain "authors" to avoid inheritance of the URL search string from the class name
    # when we don't do this, the search URL will become "https://api.openalex.org/authorswithretry?search=John+Doe"
    def _full_collection_name(self):
        # Force the endpoint to /authors
        if self.params is not None and "q" in self.params.keys():
            # If there's "q" in params, pyalex normally goes to /autocomplete/authors
            return f"{config.openalex_url}/autocomplete/authors"
        else:
            return f"{config.openalex_url}/authors"

    def get(self, max_retries=12, base_delay=2, **kwargs):
        """
        `get()` with quadratic backoff.
        """
        for attempt in range(1, max_retries + 1):
            try:
                return super().get(**kwargs)
            except (ConnectionError, ReadTimeout) as err:
                if attempt < max_retries:
                    wait_time = base_delay * (attempt ** 2)
                    print(f"[AuthorsWithRetry] Attempt {attempt} failed: {err}")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("[AuthorsWithRetry] Max retries reached. Raising error.")
                    raise


class WorksWithRetry(Works):
    """
    Wrapper around pyalex.Works that retries on ConnectionError/ReadTimeout.
    Usage example:
        works = WorksWithRetry().filter(author={"id": "https://openalex.org/A1234"}).get()
    """
    
    # Force the endpoint to remain "works" to avoid inheritance of the URL search string from the class name
    # when we don't do this, the search URL will become "https://api.openalex.org/workswithretry?filter=author.id:https%3A%2F%2Fopenalex.org%2FA1234"
    def _full_collection_name(self):
        # Force the endpoint to /works
        return f"{config.openalex_url}/works"

    def get(self, max_retries=12, base_delay=2, **kwargs):
        """
        `get()` with quadratic backoff.
        """
        for attempt in range(1, max_retries + 1):
            try:
                return super().get(**kwargs)
            except (ConnectionError, ReadTimeout) as err:
                if attempt < max_retries:
                    wait_time = base_delay * (attempt ** 2)
                    print(f"[WorksWithRetry] Attempt {attempt} failed: {err}")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("[WorksWithRetry] Max retries reached. Raising error.")
                    raise


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
        search_results = AuthorsWithRetry().search(author).get()

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
