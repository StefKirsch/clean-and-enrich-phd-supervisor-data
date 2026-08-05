import logging
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from os import getpid, path, makedirs
from time import monotonic

from pyalex import Authors, Works, config
import pandas as pd
import numpy as np
from requests.exceptions import RequestException
from sentence_transformers import util

from src.api_cache_helpers import OpenAlexDailyLimitError
from src.io_helpers import fetch_supervisors_from_pilot_dataset, remove_illegal_title_characters
from src.clean_names_helpers import format_name_to_firstname_lastname, name_sanity_check


_RUN_ID = f"{datetime.now():%Y%m%d-%H%M%S}-pid{getpid()}"
_CURRENT_RECORD = ContextVar(
    "openalex_record",
    default={
        "completed_rows": "-",
        "row_number": "-",
        "total_rows": "-",
        "record_id": "-",
    },
)


class _ExtractionContextFilter(logging.Filter):
    """Add shared correlation fields to every technical and progress entry."""

    def filter(self, record):
        record_context = _CURRENT_RECORD.get()
        record.run_id = _RUN_ID
        record.completed_rows = record_context["completed_rows"]
        record.row_number = record_context["row_number"]
        record.total_rows = record_context["total_rows"]
        record.record_id = record_context["record_id"]
        return True


class _NonTechnicalContentFilter(logging.Filter):
    """Keep domain-level search and matching content out of technical noise."""

    _TECHNICAL_MESSAGE_PREFIXES = (
        "ROW START:",
        "ROW END:",
        "ROW NETWORK ERROR:",
    )

    def filter(self, record):
        return not record.getMessage().startswith(
            self._TECHNICAL_MESSAGE_PREFIXES
        )


class AuthorRelations:
    # Class attribute shared by all instances
    # Can be overwritten at the class level to change the default for all (upcoming) instances in scope
    # Keys: supervisor name 
    # Values: supervisor OpenAlex ID
    supervisors_in_pilot_dataset = dict()
    _technical_log_handler = None
    _nontechnical_log_handler = None
    _run_started_logged = False
    
    def __init__(
        self,
        integer_id,
        phd_name,
        title,
        year,
        institution,
        contributors,
        model,
        years_tolerance=0,
        thesis_identifier=None,
    ):
        self.integer_id = integer_id
        self.thesis_identifier = thesis_identifier
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
        self.processing_error = None
        self.progress_step = 0
        self.current_progress_event = None
        self.active_contributor_name = None
        self.active_contributor_rank = None
        
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
        self.years_offset_phd_matching = [float("inf"), 1]
        
        # Define target years as a property of the object
        self.affiliation_target_years = self.calculate_affiliation_target_years()
        
        # Minimum number of shared publications required for a contributor match 
        self.n_shared_pubs_min = 1
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.progress_logger = logging.getLogger("extraction.progress")
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
        technical_formatter = logging.Formatter(
            "%(asctime)s - run=%(run_id)s - "
            "progress=%(completed_rows)s/%(total_rows)s - "
            "active_row=%(row_number)s/%(total_rows)s - id=%(record_id)s - "
            "%(name)s - %(levelname)s - %(message)s"
        )
        # This is the original author-relations log format. The handler below
        # receives only the semantic search and matching trace, while request,
        # retry, timing, and correlation diagnostics stay in extraction.log.
        nontechnical_formatter = logging.Formatter(
            "%(asctime)s - %(message)s"
        )
        context_filter = _ExtractionContextFilter()

        if self.__class__._technical_log_handler is None:
            technical_handler = RotatingFileHandler(
                "extraction.log",
                maxBytes=20 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            technical_handler.setFormatter(technical_formatter)
            technical_handler.addFilter(context_filter)
            technical_handler._author_relations_handler = True
            self.__class__._technical_log_handler = technical_handler

            nontechnical_handler = RotatingFileHandler(
                "author_relations.log",
                maxBytes=20 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            nontechnical_handler.setLevel(logging.DEBUG)
            nontechnical_handler.setFormatter(nontechnical_formatter)
            nontechnical_handler.addFilter(_NonTechnicalContentFilter())
            nontechnical_handler._author_relations_content_handler = True
            self.__class__._nontechnical_log_handler = nontechnical_handler

            # Start each run in clean active files while preserving previous
            # runs in the normal numbered rotation backups. This also moves
            # the legacy technical author_relations.log out of the active
            # human-readable file on the first run after the rename.
            for handler in (technical_handler, nontechnical_handler):
                if path.exists(handler.baseFilename) and path.getsize(handler.baseFilename):
                    handler.doRollover()
        else:
            technical_handler = self.__class__._technical_log_handler
            nontechnical_handler = self.__class__._nontechnical_log_handler

        technical_handler.setLevel(logging.DEBUG)
        nontechnical_handler.setLevel(logging.DEBUG)

        def apply_logger_file_policy(
            name,
            *,
            file_handler=None,
            level=None,
            enabled=True,
            propagate=False,
        ):
            logger = logging.getLogger(name)

            logger.propagate = propagate
            logger.disabled = not enabled

            if not enabled:
                return logger

            if level is not None:
                logger.setLevel(level)

            if file_handler is not None and file_handler not in logger.handlers:
                logger.addHandler(file_handler)

            return logger

        # Custom logger with matching diagnostics
        self.logger = apply_logger_file_policy(
            __name__,
            level=logging.DEBUG,
            file_handler=technical_handler,
        )
        if nontechnical_handler not in self.logger.handlers:
            self.logger.addHandler(nontechnical_handler)

        # PyAlex messages (if any)
        apply_logger_file_policy(
            "pyalex",
            level=logging.DEBUG,
            file_handler=technical_handler,
        )

        # urllib3 retry diagnostics
        apply_logger_file_policy(
            "urllib3.util.retry",
            level=logging.DEBUG,
            file_handler=technical_handler,
        )

        # Timed start/end/failure messages for each OpenAlex request.
        apply_logger_file_policy(
            "openalex.http",
            level=logging.INFO,
            file_handler=technical_handler,
        )

        # Synthetic progress/correlation messages are technical diagnostics.
        self.progress_logger = apply_logger_file_policy(
            "extraction.progress",
            level=logging.INFO,
            file_handler=technical_handler,
        )

        # Suppress normal successful HTTP request chatter.
        apply_logger_file_policy(
            "urllib3.connectionpool",
            enabled=False,
        )

        if not self.__class__._run_started_logged:
            self.progress_logger.info(
                "step=-- | RUN START | Detailed diagnostics: extraction.log; "
                "match entries using run, active row, id, and step."
            )
            self.__class__._run_started_logged = True

    def log_progress(self, event, message):
        """Write one correlated, human-readable step to both log files."""
        self.progress_step += 1
        self.current_progress_event = event
        self.progress_logger.info(
            "step=%02d | %s | %s",
            self.progress_step,
            event,
            message,
        )

    def search_phd_candidate(self):
        """
        Search for the PhD candidate by name and validate by name match, fuzzy match of works and institution.
        Collect all candidates into a list, then decide which best matches criteria.

        Now we also assign a 'match_score' to each candidate:
            match_score = n_close_matches + (50 if exact_match) + (20 if near_exact_match) + (20 if affiliation_match)
        Then we pick the candidate with the highest match_score.

        Log a table representation of the sorted DataFrame.
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
                'candidate_orcid': candidate['orcid'],
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


        columns_to_show = [
            'candidate_name', 'candidate_id', 'candidate_orcid', 'match_score',
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
            
    def _init_supervisor_record(self, contributor_name, rank, openalex_candidates=None):
        return {
            'contributor_name_narcis': contributor_name,
            'name_matches_open_alex': [],
            'contributor_rank': rank,
            'supervisor': openalex_candidates or [],
            'supervisor_confirmed': False,
            'same_grad_inst': False,
            'n_shared_inst_grad': 0,
            'is_sup_in_pilot_dataset': False,
            'sup_match_by': '',
            'n_shared_pubs': 0,
            'shared_pubs': [],
            'is_thesis_coauthor': False,
        }
        
    
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
        
        self.potential_supervisors = []

        # Case 1: PhD candidate not confirmed
        if not self.phd_candidate or not self.phd_match_by:
            self.logger.warning(
                "PhD candidate not confirmed. Cannot find potential supervisors; "
                "recording contributors only with NARCIS data."
            )
            self.log_progress(
                "SUPERVISOR CHECK SKIPPED",
                "The PhD candidate was not confirmed; contributor names were retained.",
            )

            for idx, contributor_name in enumerate(self.contributors):
                contributor_rank = idx + 1
                self.potential_supervisors.append(
                    self._init_supervisor_record(contributor_name, contributor_rank)
                )

            return self.potential_supervisors

        # Get PhD candidate's affiliations at graduation
        phd_affiliations_at_graduation = self.get_candidate_affiliations(
            self.phd_candidate, in_target_years=True
        )
        
        # Case 2: no affiliations in target years
        if not phd_affiliations_at_graduation:
            self.logger.warning(
                "PhD candidate has no affiliations in target years. Cannot find potential supervisors; "
                "recording contributors only with NARCIS data."
            )
            self.log_progress(
                "SUPERVISOR CHECK SKIPPED",
                "No affiliation was found around graduation; contributor names were retained.",
            )

            for idx, contributor_name in enumerate(self.contributors):
                contributor_rank = idx + 1
                self.potential_supervisors.append(
                    self._init_supervisor_record(contributor_name, contributor_rank)
                )

            return self.potential_supervisors

        # Case 3: PhD candidate confirmed with affiliation in target years
        self.logger.debug(
            f"Target Institutions: {phd_affiliations_at_graduation}, "
            f"Target Years: {self.affiliation_target_years}"
        )
        self.logger.info("Searching for potential supervisors among contributors.")

        self.potential_supervisors = []
        phd_dois = self.phd_publications["doi"].tolist()

        for idx, contributor_name in enumerate(self.contributors):
            contributor_rank = idx + 1
            self.active_contributor_name = contributor_name
            self.active_contributor_rank = contributor_rank
            self.logger.debug(f"Processing contributor #{contributor_rank}: {contributor_name}")
            self.log_progress(
                "CONTRIBUTOR START",
                f"{contributor_rank}/{len(self.contributors)} | name={contributor_name!r}",
            )

            # Search for contributors in OpenAlex
            openalex_candidates = Authors().search(contributor_name).get()
            self.logger.debug(
                f"Found {len(openalex_candidates)} OpenAlex candidates for '{contributor_name}'."
            )
            
            # Allocate dict for aggregated supervisor data
            supervisor_data = self._init_supervisor_record(
                contributor_name=contributor_name,
                rank=contributor_rank,
                openalex_candidates=openalex_candidates,
            )

            if not openalex_candidates:
                self.logger.debug(
                    f"No OpenAlex matches for '{contributor_name}'. Adding placeholder entry."
                )
                self.potential_supervisors.append(supervisor_data)
                self.log_progress(
                    "CONTRIBUTOR END",
                    f"{contributor_rank}/{len(self.contributors)} | "
                    f"name={contributor_name!r} | supervisor confirmed=no",
                )
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
            # criteria_met = coauthorship_flag # We require at least one shared publication.
            # sup_match_by = f"Name match and ≥ {self.n_shared_pubs_min} shared publication(s)."
            
            # criteria_met = same_grad_inst_flag # We require that both are affiliated with the institution in NARCIS at graduation
            # sup_match_by = "Affiliated with NARCIS institution at graduation."
            
            criteria_met = bool(len(all_shared_affils)) # We require at least one shared affiliation in the target years around graduation
            sup_match_by = "Shared affiliation at graduation."
            
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
            self.log_progress(
                "CONTRIBUTOR END",
                f"{contributor_rank}/{len(self.contributors)} | "
                f"name={contributor_name!r} | "
                f"supervisor confirmed={'yes' if criteria_met else 'no'}",
            )

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
        also return a single-row DataFrame with 'phd_name', 'phd_id', 'phd_orcid', 'phd_match_by' filled
        and all supervisor-related columns as NaN.
        """
        
        # The columns our DataFrame should have
        columns = [
            'integer_id',
            'thesis_identifier',
            'institution',
            'phd_name', 
            'phd_id', 
            'phd_orcid',
            'n_name_search_matches',
            'duplicate_phds',
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
            'contributor_orcid',
            'sup_match_by',
            'contributor_rank', 
            'same_grad_inst', 
            'n_shared_inst_grad', 
            'is_sup_in_pilot_dataset', 
            'n_shared_pubs', 
            'shared_pubs', 
            'is_thesis_coauthor',
            'processing_error',
        ]

        integer_id = self.integer_id
        phd_id = self.phd_candidate['id'] if self.phd_candidate else None
        phd_orcid = self.phd_candidate['orcid'] if self.phd_candidate else None
        phd_name = self.phd_candidate['display_name'] if self.phd_candidate else self.phd_name
        title_open_alex = self.title_open_alex if self.title_open_alex else self.title # convert empty list to None
        title_similarities = self.title_similarities or None # convert empty list to None
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
            contributor_orcid = [supervisor_nested['orcid'] for supervisor_nested in supervisor]
            sup_match_by = supervisor_data['sup_match_by']
            contributor_rank = supervisor_data['contributor_rank']
            same_grad_inst = supervisor_data['same_grad_inst']
            n_shared_inst_grad = supervisor_data['n_shared_inst_grad']
            is_sup_in_pilot_dataset = supervisor_data['is_sup_in_pilot_dataset']
            shared_pubs = supervisor_data['shared_pubs']
            is_thesis_coauthor = supervisor_data['is_thesis_coauthor']

            result_row = {
                'integer_id': integer_id,
                'thesis_identifier': self.thesis_identifier,
                'institution': self.institution,
                'phd_name': phd_name,
                'phd_id': phd_id,
                'phd_orcid': phd_orcid,
                'n_name_search_matches': self.n_name_search_matches,
                'duplicate_phds': None,
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
                'contributor_orcid': contributor_orcid,
                'sup_match_by': sup_match_by,
                'contributor_rank': contributor_rank,
                'same_grad_inst': same_grad_inst,
                'n_shared_inst_grad': n_shared_inst_grad,
                'is_sup_in_pilot_dataset': is_sup_in_pilot_dataset,
                'n_shared_pubs': len(shared_pubs),
                'shared_pubs': shared_pubs,
                'is_thesis_coauthor': is_thesis_coauthor,
                'processing_error': self.processing_error,
            }
            results_list.append(result_row)
        
        if not results_list:
            if self.phd_candidate:
                self.logger.warning("PhD candidate confirmed, but no supervisors found.")
            else:
                self.logger.warning("No confirmed PhD candidate or supervisors found.")
            # Create a single row with the data we have and the others as None
            result_row = {col: None for col in columns}
            result_row['integer_id'] = integer_id
            result_row['thesis_identifier'] = self.thesis_identifier
            result_row['institution'] = self.institution
            result_row['phd_name'] = phd_name
            result_row['phd_id'] = phd_id
            result_row['phd_orcid'] = phd_orcid
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
            result_row['processing_error'] = self.processing_error
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


def remove_duplicate_phd_candidates(extraction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate PhD candidates with different integer IDs.

    Prefer ORCIDs for identifying candidates and use OpenAlex IDs when no ORCID
    is available. For duplicates, keep the integer ID with the most confirmed
    contributors. Ties are resolved by keeping the first integer ID. Add all
    duplicate source-record metadata, PhD names, and OpenAlex IDs to the
    retained candidate for diagnostics.
    """
    candidates = (
        extraction_df.drop_duplicates(subset="integer_id").copy().reset_index(drop=True)
    )
    candidates["duplicate_id"] = (
        candidates["phd_orcid"].replace("", pd.NA).fillna(candidates["phd_id"])
    )
    candidates["n_confirmed_contributors"] = candidates["integer_id"].map(
        extraction_df.groupby("integer_id", sort=False)["contributor_confirmed"].sum()
    )

    candidates_with_id = candidates[candidates["duplicate_id"].notna()]
    duplicate_integer_ids = []
    duplicate_phds_by_integer_id = {}

    for _, duplicate_candidates in candidates_with_id.groupby("duplicate_id", sort=False):
        if len(duplicate_candidates) == 1:
            continue

        kept_index = duplicate_candidates["n_confirmed_contributors"].idxmax()
        kept_candidate = duplicate_candidates.loc[[kept_index]]
        removed_candidates = duplicate_candidates.drop(index=kept_index)
        ordered_candidates = pd.concat([kept_candidate, removed_candidates])
        kept_integer_id = kept_candidate["integer_id"].iloc[0]

        duplicate_integer_ids.extend(removed_candidates["integer_id"])
        duplicate_phds_by_integer_id[kept_integer_id] = (
            ordered_candidates[
                [
                    "integer_id",
                    "thesis_identifier",
                    "institution",
                    "phd_name",
                    "phd_id",
                ]
            ].to_dict("records")
        )

    result = extraction_df[
        ~extraction_df["integer_id"].isin(duplicate_integer_ids)
    ].copy()
    duplicate_phds = pd.Series(
        [
            duplicate_phds_by_integer_id.get(integer_id)
            for integer_id in result["integer_id"]
        ],
        index=result.index,
        dtype=object
    )

    if "duplicate_phds" in result.columns:
        result["duplicate_phds"] = duplicate_phds
    else:
        column_index = (
            result.columns.get_loc("n_name_search_matches") + 1
            if "n_name_search_matches" in result.columns
            else len(result.columns)
        )
        result.insert(column_index, "duplicate_phds", duplicate_phds)

    n_removed_candidates = len(duplicate_integer_ids)
    candidate_label = "candidate" if n_removed_candidates == 1 else "candidates"
    print(f"Removed {n_removed_candidates} duplicate PhD {candidate_label}.")

    return result


def find_phd_and_supervisors_in_row(
    row,
    model,
    row_number=None,
    total_rows=None,
):
    """
    Finds author relations information from a DataFrame row.

    Processes the row to find the PhD candidate and potential supervisors,
    and returns a DataFrame per row with the required columns.

    Parameters:
        row (pd.Series): A row from the DataFrame containing publication data.
        model: The model used to validate author matches.
        row_number (int, optional): The one-based position of the active row.
        total_rows (int, optional): The total number of rows being processed.

    Returns:
        pd.DataFrame: A DataFrame with columns as specified.
    """
    # Extract necessary fields
    integer_id = row['integer_id']
    thesis_identifier = row['thesis_identifier']
    phd_name = row['phd_name']
    title = row['title']
    year = int(row['year'])
    institution = row['institution']
    contributors = [row[f'contributor_{i}'] for i in range(1, 11) if pd.notna(row.get(f'contributor_{i}', None))]
    
    record_context_token = _CURRENT_RECORD.set(
        {
            "completed_rows": (
                row_number - 1
                if isinstance(row_number, int)
                else "-"
            ),
            "row_number": row_number if row_number is not None else "-",
            "total_rows": total_rows if total_rows is not None else "-",
            "record_id": integer_id,
        }
    )
    try:
        # Create an instance of AuthorRelations.
        author_relations = AuthorRelations(
            integer_id=integer_id,
            phd_name=phd_name,
            title=title,
            year=year,
            institution=institution,
            contributors=contributors,
            model=model,
            years_tolerance=-4, # cf. issue #19
            thesis_identifier=thesis_identifier,
        )

        started_at = monotonic()
        author_relations.log_progress(
            "RECORD START",
            f"PhD={phd_name!r} | year={year} | "
            f"listed contributors={len(contributors)}",
        )
        author_relations.logger.info(
            "ROW START: integer_id=%s phd_name=%r year=%s contributors=%s",
            integer_id,
            phd_name,
            year,
            len(contributors),
        )

        try:
            author_relations.log_progress(
                "PHD SEARCH START",
                f"Searching for PhD candidate {phd_name!r}.",
            )
            author_relations.search_phd_candidate()
            matched_phd_name = (
                author_relations.phd_candidate["display_name"]
                if author_relations.phd_candidate
                else None
            )
            author_relations.log_progress(
                "PHD SEARCH END",
                (
                    f"Candidate confirmed=yes | matched name={matched_phd_name!r}"
                    if author_relations.phd_match_by
                    else "Candidate confirmed=no"
                ),
            )

            author_relations.log_progress(
                "SUPERVISOR CHECK START",
                f"Checking {len(contributors)} listed contributor(s).",
            )
            author_relations.collect_supervision_metadata()
            confirmed_supervisors = sum(
                1
                for supervisor in author_relations.potential_supervisors
                if supervisor["supervisor_confirmed"]
            )
            author_relations.log_progress(
                "SUPERVISOR CHECK END",
                f"Confirmed supervisors={confirmed_supervisors}.",
            )
        except OpenAlexDailyLimitError:
            # A daily quota cannot recover on the next row. Let the top-level
            # extraction stop before it writes a plausible-looking but
            # incomplete output dataset.
            raise
        except RequestException as exc:
            # Preserve the row and continue with the dataset after bounded
            # retries are exhausted. Technical details stay out of the
            # human-readable progress log.
            author_relations.processing_error = f"{type(exc).__name__}: {exc}"

            error_type = type(exc).__name__
            if author_relations.current_progress_event == "PHD SEARCH START":
                skipped_content_message = (
                    "The OpenAlex request did not complete after retries during the PhD "
                    f"candidate search. Skipped PhD matching and all "
                    f"{len(contributors)} listed contributor checks for this "
                    "record; a placeholder or partial record was retained."
                )
                skipped_data_message = (
                    f"OpenAlex API error after retries ({error_type}) during the "
                    f"PhD candidate search. PhD matching and all listed "
                    f"contributor checks ({len(contributors)}) for this record "
                    "were skipped; a placeholder or partial record was retained."
                )
            elif author_relations.active_contributor_rank is not None:
                contributor_rank = author_relations.active_contributor_rank
                later_contributors = len(contributors) - contributor_rank
                if later_contributors:
                    contributor_label = (
                        "contributor"
                        if later_contributors == 1
                        else "contributors"
                    )
                    skipped_contributors = (
                        f"Data for this contributor and the {later_contributors} "
                        f"later {contributor_label} was skipped"
                    )
                else:
                    skipped_contributors = "Data for this contributor was skipped"
                skipped_content_message = (
                    "The OpenAlex request did not complete after retries while checking "
                    f"contributor {contributor_rank}/{len(contributors)} "
                    f"{author_relations.active_contributor_name!r}. "
                    f"{skipped_contributors}; already completed data for this "
                    "record was retained."
                )
                skipped_data_message = (
                    f"OpenAlex API error after retries ({error_type}) while "
                    f"checking contributor {contributor_rank}/{len(contributors)} "
                    f"{author_relations.active_contributor_name!r}. "
                    f"{skipped_contributors}; already completed data for this "
                    "record was retained."
                )
            else:
                skipped_content_message = (
                    "The OpenAlex request did not complete after retries during supervisor "
                    "checking. Skipped unfinished supervisor data; already "
                    "completed data for this record was retained."
                )
                skipped_data_message = (
                    f"OpenAlex API error after retries ({error_type}) during "
                    "supervisor checking. Unfinished supervisor data was skipped; "
                    "already completed data for this record was retained."
                )

            author_relations.logger.warning(skipped_content_message)
            author_relations.log_progress(
                "NETWORK ERROR",
                skipped_data_message,
            )
            author_relations.logger.exception(
                "ROW NETWORK ERROR: integer_id=%s phd_name=%r",
                integer_id,
                phd_name,
            )

        # Get the DataFrame results.
        results_df = author_relations.get_results()
        elapsed = monotonic() - started_at
        status = (
            "completed with network error"
            if author_relations.processing_error
            else "completed"
        )
        author_relations.log_progress(
            "RECORD END",
            f"PhD={phd_name!r} | status={status} | elapsed={elapsed:.1f}s",
        )
        author_relations.logger.info(
            "ROW END: integer_id=%s phd_name=%r elapsed=%.1fs error=%s",
            integer_id,
            phd_name,
            elapsed,
            author_relations.processing_error is not None,
        )

        return results_df
    finally:
        _CURRENT_RECORD.reset(record_context_token)
    

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
    supervisors_std = [format_name_to_firstname_lastname(name) for name in supervisors]
    
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
