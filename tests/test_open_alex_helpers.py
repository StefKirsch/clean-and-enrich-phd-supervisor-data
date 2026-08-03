import unittest

import pandas as pd

from src.open_alex_helpers import remove_duplicate_phd_candidates


class RemoveDuplicatePhdCandidatesTests(unittest.TestCase):
    def test_keeps_most_confirmed_contributors(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 1, "phd_name": "One", "phd_id": "A1", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": True},
            {"integer_id": 2, "phd_name": "Two", "phd_id": "A2", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": True},
            {"integer_id": 2, "phd_name": "Two", "phd_id": "A2", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": True},
            {"integer_id": 3, "phd_name": "Three", "phd_id": "A3", "phd_orcid": "O1", "n_name_search_matches": 1, "contributor_confirmed": False},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [2, 2])
        self.assertEqual(
            result["duplicate_phds"].iloc[0],
            [
                {"phd_name": "Two", "phd_id": "A2"},
                {"phd_name": "One", "phd_id": "A1"},
                {"phd_name": "Three", "phd_id": "A3"},
            ]
        )
        self.assertEqual(
            result.columns.get_loc("duplicate_phds"),
            result.columns.get_loc("n_name_search_matches") + 1
        )

    def test_keeps_first_on_tie(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 2, "phd_name": "Two", "phd_id": "A1", "phd_orcid": "O1", "contributor_confirmed": True},
            {"integer_id": 1, "phd_name": "One", "phd_id": "A1", "phd_orcid": "O1", "contributor_confirmed": True},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [2])
        self.assertEqual(
            result["duplicate_phds"].iloc[0],
            [
                {"phd_name": "Two", "phd_id": "A1"},
                {"phd_name": "One", "phd_id": "A1"},
            ]
        )

    def test_uses_openalex_id_as_backup(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 1, "phd_name": "One", "phd_id": "A1", "phd_orcid": None, "contributor_confirmed": False},
            {"integer_id": 2, "phd_name": "Two", "phd_id": "A1", "phd_orcid": None, "contributor_confirmed": True},
            {"integer_id": 3, "phd_name": "Three", "phd_id": None, "phd_orcid": None, "contributor_confirmed": False},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [2, 3])
        self.assertIsNone(result.loc[result["integer_id"] == 3, "duplicate_phds"].iloc[0])

    def test_prefers_orcid_over_openalex_id(self):
        extraction_df = pd.DataFrame([
            {"integer_id": 1, "phd_name": "One", "phd_id": "A1", "phd_orcid": "O1", "contributor_confirmed": True},
            {"integer_id": 2, "phd_name": "Two", "phd_id": "A1", "phd_orcid": "O2", "contributor_confirmed": True},
        ])

        result = remove_duplicate_phd_candidates(extraction_df)

        self.assertEqual(result["integer_id"].tolist(), [1, 2])
        self.assertEqual(result["duplicate_phds"].tolist(), [None, None])


if __name__ == "__main__":
    unittest.main()
