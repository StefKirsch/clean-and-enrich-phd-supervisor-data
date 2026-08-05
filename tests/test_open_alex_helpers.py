import unittest
from unittest.mock import Mock

import pandas as pd

from src.open_alex_helpers import AuthorRelations, remove_duplicate_phd_candidates


class AuthorRelationsResultTests(unittest.TestCase):
    def test_keeps_source_identifiers_in_their_relative_column_positions(self):
        author_relations = AuthorRelations(
            integer_id=42,
            thesis_identifier="thesis:42",
            institution="Example University",
            phd_name="Ada Example",
            title="Example thesis",
            year=2020,
            contributors=[],
            model=Mock(),
        )

        result = author_relations.get_results()

        self.assertEqual(
            result.columns[:4].tolist(),
            ["integer_id", "thesis_identifier", "institution", "phd_name"],
        )
        self.assertEqual(result.loc[0, "thesis_identifier"], "thesis:42")
        self.assertEqual(result.loc[0, "institution"], "Example University")


class RemoveDuplicatePhdCandidatesTests(unittest.TestCase):
    def test_keeps_most_confirmed_contributors(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 1, "thesis_identifier": "T1", "institution": "I1", "phd_name": "One", "phd_id": "A1", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": True},
            {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A2", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": True},
            {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A2", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": True},
            {"integer_id": 3, "thesis_identifier": "T3", "institution": "I3", "phd_name": "Three", "phd_id": "A3", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": False},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [2, 2])
        self.assertEqual(
            result["duplicate_phds"].iloc[0],
            [
                {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A2"},
                {"integer_id": 1, "thesis_identifier": "T1", "institution": "I1", "phd_name": "One", "phd_id": "A1"},
                {"integer_id": 3, "thesis_identifier": "T3", "institution": "I3", "phd_name": "Three", "phd_id": "A3"},
            ]
        )
        self.assertEqual(
            result.columns.get_loc("duplicate_phds"),
            result.columns.get_loc("n_name_search_matches") + 1
        )

    def test_keeps_first_on_tie(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A1", "phd_orcid": "O1", "contributor_confirmed": True},
            {"integer_id": 1, "thesis_identifier": "T1", "institution": "I1", "phd_name": "One", "phd_id": "A1", "phd_orcid": "O1", "contributor_confirmed": True},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [2])
        self.assertEqual(
            result["duplicate_phds"].iloc[0],
            [
                {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A1"},
                {"integer_id": 1, "thesis_identifier": "T1", "institution": "I1", "phd_name": "One", "phd_id": "A1"},
            ]
        )

    def test_uses_openalex_id_as_backup(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 1, "thesis_identifier": "T1", "institution": "I1", "phd_name": "One", "phd_id": "A1", "phd_orcid": None, "contributor_confirmed": False},
            {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A1", "phd_orcid": None, "contributor_confirmed": True},
            {"integer_id": 3, "thesis_identifier": "T3", "institution": "I3", "phd_name": "Three", "phd_id": None, "phd_orcid": None, "contributor_confirmed": False},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [2, 3])
        self.assertIsNone(result.loc[result["integer_id"] == 3, "duplicate_phds"].iloc[0])

    def test_prefers_orcid_over_openalex_id(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 1, "thesis_identifier": "T1", "institution": "I1", "phd_name": "One", "phd_id": "A1", "phd_orcid": "O1", "contributor_confirmed": True},
            {"integer_id": 2, "thesis_identifier": "T2", "institution": "I2", "phd_name": "Two", "phd_id": "A1", "phd_orcid": "O2", "contributor_confirmed": True},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [1, 2])
        self.assertEqual(result["duplicate_phds"].tolist(), [None, None])


if __name__ == "__main__":
    unittest.main()
