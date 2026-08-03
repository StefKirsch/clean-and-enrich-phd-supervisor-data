import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Test how duplicates are handled based on known cases.
# Run this test like so
# `$env:RUN_OPENALEX_INTEGRATION="1"; .\.venv\Scripts\python.exe -m unittest tests.test_open_alex_match_integration -v

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_IDS = {2127, 99128, 97227, 414, 38061, 43235}
DUPLICATE_PAIRS = (
    {2127, 99128},
    {97227, 414},
    {38061, 43235}, # Complete duplicate other that institution. Is now removed during data cleaning, so it's not in the cleaned publications anymore


@unittest.skipUnless(
    os.environ.get("RUN_OPENALEX_INTEGRATION") == "1",
    "Set RUN_OPENALEX_INTEGRATION=1 to run the live OpenAlex integration test.",
)
class OpenAlexMatchIntegrationTests(unittest.TestCase):
    def test_full_script_removes_confirmed_duplicate_candidates(self):
        project_config = yaml.safe_load(
            (PROJECT_ROOT / "dataset_config.yaml").read_text(encoding="utf-8")
        )
        source_path = Path(project_config["dataset_path"])
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        self.assertTrue(
            source_path.is_file(),
            f"Full integration dataset not found: {source_path}",
        )

        full_input = pd.read_csv(source_path, low_memory=False)
        filtered_input = full_input[
            full_input["integer_id"].isin(INTEGRATION_IDS)
        ].copy()
        found_ids = set(filtered_input["integer_id"])
        partially_present_pairs = [
            pair for pair in DUPLICATE_PAIRS if pair & found_ids and not pair <= found_ids
        ]
        self.assertFalse(
            partially_present_pairs,
            "The current full dataset contains only one candidate from these "
            f"duplicate pairs: {partially_present_pairs}",
        )
        tested_pairs = tuple(pair for pair in DUPLICATE_PAIRS if pair <= found_ids)
        self.assertTrue(
            tested_pairs,
            "None of the requested duplicate pairs occur in the current full dataset.",
        )
        self.assertEqual(len(filtered_input), len(found_ids))

        with tempfile.TemporaryDirectory(
            prefix="openalex-duplicate-integration-",
        ) as temporary_directory:
            test_directory = Path(temporary_directory)
            input_path = test_directory / "duplicate_candidates.csv"
            output_path = test_directory / "author_relations.csv"
            config_path = test_directory / "dataset_config.yaml"
            filtered_input.to_csv(input_path, index=False)

            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset_path": str(input_path),
                        "dataset_path_sample_gold_standard": None,
                        "output_filename": str(output_path),
                        "output_filename_gold_standard": None,
                        "use_sample_for_gold_standard": False,
                        "domain": None,
                        "chunk": None,
                        "nrows": None,
                        "min_rank_contrib": None,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            pilot_source = PROJECT_ROOT / "data" / "output" / "sups_pilot.csv"
            if pilot_source.is_file():
                pilot_target = test_directory / "data" / "output" / "sups_pilot.csv"
                pilot_target.parent.mkdir(parents=True)
                shutil.copy2(pilot_source, pilot_target)

            for secret_name in ("contact_email.txt", "openalex_api_key.txt"):
                secret_source = PROJECT_ROOT / secret_name
                if secret_source.is_file():
                    shutil.copy2(secret_source, test_directory / secret_name)

            cache_source = PROJECT_ROOT / "openalex_cache.sqlite"
            if cache_source.is_file():
                os.link(cache_source, test_directory / "openalex_cache.sqlite")

            environment = os.environ.copy()
            environment.update(
                {
                    "MPLBACKEND": "Agg",
                    "OPENALEX_DATASET_CONFIG": str(config_path),
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "Open_Alex_Match.py")],
                cwd=test_directory,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=1800,
            )

            artifact_root = Path(
                os.environ.get(
                    "OPENALEX_TEST_ARTIFACT_DIR",
                    PROJECT_ROOT / ".test-artifacts" / "openalex-match",
                )
            )
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            artifact_directory = artifact_root / run_id
            artifact_directory.mkdir(parents=True, exist_ok=False)

            if output_path.is_file():
                shutil.copy2(
                    output_path,
                    artifact_directory / "author_relations.csv",
                )
            shutil.copy2(
                config_path,
                artifact_directory / "dataset_config.yaml",
            )
            (artifact_directory / "stdout.txt").write_text(
                completed.stdout,
                encoding="utf-8",
            )
            (artifact_directory / "stderr.txt").write_text(
                completed.stderr,
                encoding="utf-8",
            )
            print(f"Integration-test artifacts: {artifact_directory}")

            self.assertEqual(
                completed.returncode,
                0,
                "Open_Alex_Match.py failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}",
            )
            result = pd.read_csv(output_path)
            candidate_summary = result.drop_duplicates("integer_id")[
                [
                    "integer_id",
                    "phd_name",
                    "phd_id",
                    "phd_orcid",
                    "phd_match_by",
                    "duplicate_phds",
                ]
            ].to_string(index=False)

            retained_ids = set(result["integer_id"])
            pairs_with_missing_matches = 0

            for duplicate_pair in tested_pairs:
                retained_from_pair = retained_ids & duplicate_pair
                if len(retained_from_pair) == 2:
                    pair_match_by = result.loc[
                        result["integer_id"].isin(duplicate_pair),
                        "phd_match_by",
                    ]
                    self.assertTrue(
                        pair_match_by.isna().any(),
                        f"Expected pair {sorted(duplicate_pair)} to be deduplicated "
                        "because neither candidate has a missing phd_match_by value.\n"
                        f"{candidate_summary}",
                    )
                    pairs_with_missing_matches += 1
                    continue

                self.assertEqual(
                    len(retained_from_pair),
                    1,
                    f"Expected one retained candidate from {sorted(duplicate_pair)}, "
                    "or both candidates when phd_match_by is missing; "
                    f"found {sorted(retained_from_pair)}.",
                )
                retained_id = retained_from_pair.pop()
                duplicate_values = (
                    result.loc[
                        result["integer_id"] == retained_id,
                        "duplicate_phds",
                    ]
                    .dropna()
                    .unique()
                )
                self.assertEqual(len(duplicate_values), 1)
                duplicate_metadata = ast.literal_eval(duplicate_values[0])
                self.assertEqual(len(duplicate_metadata), 2)
                self.assertTrue(
                    all(entry["phd_id"] for entry in duplicate_metadata)
                )

            expected_removed_candidates = len(tested_pairs) - pairs_with_missing_matches
            expected_candidate_label = (
                "candidate" if expected_removed_candidates == 1 else "candidates"
            )
            self.assertIn(
                f"Removed {expected_removed_candidates} duplicate PhD "
                f"{expected_candidate_label}.",
                completed.stdout,
                f"Unexpected duplicate-removal count.\n{candidate_summary}",
            )
            self.assertEqual(
                len(retained_ids),
                len(tested_pairs) + pairs_with_missing_matches,
            )


if __name__ == "__main__":
    unittest.main()
