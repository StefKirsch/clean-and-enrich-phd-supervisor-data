# %% [markdown]
# # Open Alex Extraction and Matching with .search()
# 
# The goal of this Notebook is look up the PhD students (Authors) contained in the [cleaned](clean_data.ipynb) NARCIS dataset, and
# 1. Confirm they can be found in OpenAlex
# 2. Confirm their affiliation in NARCIS matches the one in OpenAlex
# 2. Confirm they wrote the associated PhD Thesis
# 3. Per author, look up all the contributors (i.e. potential first supervisors) that are listen in the NARCIS dataset
# 
# The previous version of this notebook written by a Bachelor student was using the `.search_filter()` method of `pyalex`, which does not search alternate spellings of the specified name. In this notebook we are using `search_filter()`, which does not have that problem. See the example code [here](search_parameter_vs_search_filter.ipynb).

# %% [markdown]
# ## 1. Setup

# %%
#from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders, Concepts
from pyalex import config # to set email_address
import pandas as pd
from sentence_transformers import SentenceTransformer
from os import path
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

from src.unabbreviate_institutions import unabbreviate_institutions
from src.open_alex_helpers import AuthorRelations, find_phd_and_supervisors_in_row, get_supervisors_openalex_ids
from src.dataset_config_helpers import read_config, load_dataset
from src.api_cache_helpers import (
    OpenAlexDailyLimitError,
    configure_pyalex_http_timeout,
    ensure_openalex_budget_available,
    initialize_request_cache,
)
from src.plotters import PhDMatchPlotter, ContributorMatchPlotter

# Initialize tqdm for progress bars
tqdm.pandas(
    file=sys.stdout,
    ncols=100,
    miniters=1,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# Install the cache before any API calls are made.
# This will cache every API call to Open Alex and if a cached version of the call is available,
# it will be preferred over making a new API call.
initialize_request_cache()

# %% [markdown]
# Notebook settings

# limit the number of rows that are shown with printing data frames
pd.set_option('display.max_rows', 5)

# %% [markdown]
# Set contact email address to get to use the [polite pool](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication#the-polite-pool). Also, if you are on a premium plan, you can access the higher usage limit by using the associated email address.

# %%
# Get contact email address and api key from files
def read_secret(filename):
    if path.isfile(filename):
        with open(filename) as f:
            return f.read().strip()

config.email = read_secret("contact_email.txt")
config.api_key = read_secret("openalex_api_key.txt")

# %% [markdown]
# Configure number of retries and backoff factor
# Pyalex is using [urllib3.util.Retry](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html) for retrying.

# %%
config.max_retries = 3
config.retry_backoff_factor = 1
configure_pyalex_http_timeout(
    connect_timeout=5,
    read_timeout=20,
    respect_retry_after=True,
    max_retry_backoff=10,
)

# Stop before loading the model or processing rows if OpenAlex cannot accept
# any new billable requests today. If this lightweight check is unavailable,
# extraction may continue; runtime 429 handling still protects the output.
try:
    rate_limit_status = ensure_openalex_budget_available(config.api_key)
except OpenAlexDailyLimitError as exc:
    raise SystemExit(f"Extraction not started: {exc}") from None
except Exception as exc:
    print(f"WARNING: OpenAlex budget preflight unavailable: {exc}")
else:
    if rate_limit_status:
        print(
            "OpenAlex API budget: "
            f"${rate_limit_status['daily_remaining_usd']} daily and "
            f"${rate_limit_status['prepaid_remaining_usd']} prepaid remaining; "
            f"resets at {rate_limit_status['resets_at']}."
        )

# %% [markdown]
# ## 2. Load datasets

# %% [markdown]
# ### 2.1 Cleaned processed NARCIS dataset

# %%
# Read configuration, including information on which subset of the data to use
data_config = read_config('dataset_config.yaml')

# Get the file name for the output file
output_filename = data_config['output_filename'] or None

# Rank after which we cut of contributors.
min_rank_contrib = data_config['min_rank_contrib'] or None
if min_rank_contrib:
    print(f"Considering the first {min_rank_contrib} contributors per PhD candidate.")
else:
    print(f"Considering all contributors.")

pubs_df = load_dataset(config=data_config)

pubs_df

# %%
# replace institution abbreviation with names that can be found in OpenAlex
# drop exact duplicates after this step
pubs_unabbrev_df = unabbreviate_institutions(pubs_df, 'institution').drop_duplicates()
pubs_unabbrev_df

# %% [markdown]
# ### 2.2 Handling several contributors per PhD

# %% [markdown]
# PhD candidates with 4 or more supervisors (for information).

# %%
min_n_contributors_to_flag = 4
contributor_cols = pubs_unabbrev_df.filter(like="contributor_").columns

# Count non-missing contributor entries per row
pubs_unabbrev_df['contributor_count'] = pubs_unabbrev_df[contributor_cols].notna().sum(axis=1)

# Reorder columns to place contributor_count after institution
cols = list(pubs_unabbrev_df.columns)
if 'institution' in cols and 'contributor_count' in cols:
    cols.remove('contributor_count')
    institution_index = cols.index('institution')
    cols.insert(institution_index + 1, 'contributor_count')
    pubs_unabbrev_df = pubs_unabbrev_df[cols]

# Filter and sort
pubs_more_than_n_df = (
    pubs_unabbrev_df[pubs_unabbrev_df['contributor_count'] >= min_n_contributors_to_flag]
    .sort_values(by=['institution', 'contributor_count'], ascending=[True, True])
    .copy()
)

print(f"There are {pubs_more_than_n_df.shape[0]} PhD candidates with 4 or more supervisors")
    
pubs_more_than_n_df

# %% [markdown]
# Remove lower rank contributors, c.f. [#49](https://github.com/StefKirsch/clean-and-enrich-phd-supervisor-data/issues/49).

# %%
pubs_high_contrib_df = pubs_unabbrev_df.copy()

# Get low ranking contributor columns 
if min_rank_contrib:
    low_rank_contrib_cols = [
        col for col in contributor_cols 
        if int(col.split('_')[1]) > min_rank_contrib
    ]

    pubs_high_contrib_df[low_rank_contrib_cols] = pd.NA

pubs_high_contrib_df


# %% [markdown]
# ### 2.3 Priority supervisor list from ResponsibleSupervision pilot
# 
# This dataset was created during the Responsible Supervision pilot project, see [here](https://github.com/tamarinde/ResponsibleSupervision/tree/main/Pilot-responsible-supervision).

# %%
repo_url = "https://github.com/tamarinde/ResponsibleSupervision/tree/main/Pilot-responsible-supervision/data/spreadsheets"
csv_path = "data/output/sups_pilot.csv"

try:
    # Attempt to read the supervisors in the pilot dataset from csv_path
    # If it fails, we get them again from GitHub
    supervisors_in_pilot_dataset = get_supervisors_openalex_ids(repo_url, csv_path)
    print(f"There are {len(supervisors_in_pilot_dataset)} Unique Supervisors with OpenAlex IDs.")
except Exception as e:
    print(f"An error occurred: {e}")

# %% [markdown]
# ## 3. Extraction

# %% [markdown]
# Load the pre-trained SPECTER model by allenai (designed for scientific documents). We pre-load the model here, so that we don't need to do that per class instance.
# 
# Citation information can be found here: https://github.com/allenai/specter

# %%
model = SentenceTransformer("allenai-specter")

# %%
# set the dict to overwrite the default class attribute specified in src/open_alex_helpers.py
AuthorRelations.supervisors_in_pilot_dataset = supervisors_in_pilot_dataset

print(
    r"""
    ╭──────────────────────────────────────────────╮
    │  Starting extraction from OpenAlex            │
    ╰──────────────────────────────────────────────╯

    Listen to the data log:
    Get-Content .\author_relations.log -Tail 30 -Wait

    Listen to the technical connection log:
    Get-Content .\extraction.log -Tail 30 -Wait
    """
)
# Process rows explicitly so the progress bar shows which record is active.
extraction_frames = []
entries_with_skipped_data = 0
with tqdm(
    total=len(pubs_high_contrib_df),
    file=sys.stdout,
    ncols=120,
    miniters=1,
    desc="Matching",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
) as progress:
    for row_number, (_, row) in enumerate(pubs_high_contrib_df.iterrows(), start=1):
        progress.set_postfix_str(
            f"row={row_number}, id={row['integer_id']}, phd={str(row['phd_name'])[:30]}",
            refresh=True,
        )
        try:
            row_result = find_phd_and_supervisors_in_row(
                row,
                model,
                row_number=row_number,
                total_rows=len(pubs_high_contrib_df),
            )
        except OpenAlexDailyLimitError as exc:
            raise SystemExit(
                "Extraction stopped before writing output: "
                f"{exc} Completed rows in this run: {row_number - 1}."
            ) from None
        extraction_frames.append(row_result)
        if row_result["processing_error"].notna().any():
            entries_with_skipped_data += 1
        progress.update()

# Concatenate all per-PhD results into one DataFrame.
extraction_df = pd.concat(extraction_frames, ignore_index=True)

extraction_df.to_csv(output_filename, index=False)

if entries_with_skipped_data:
    print(
        f"WARNING: Data was skipped for {entries_with_skipped_data} of "
        f"{len(pubs_high_contrib_df)} entries because OpenAlex requests "
        "failed after retries. See author_relations.log for what was skipped "
        "and extraction.log for technical details."
    )
else:
    print(
        f"Full extraction completed successfully: all "
        f"{len(pubs_high_contrib_df)} entries were processed without skipped data."
    )

extraction_df

# %% [markdown]
# ### Handle duplicate PhDs

# %%
dups = extraction_df[extraction_df.duplicated(subset=['phd_id'], keep=False)].sort_values(by='phd_name')

dups

# %% [markdown]
# ## 4. Analysis and Visualization

# %% [markdown]
# Load the extraction dataset from file in case we didn't run the extraction

# %%
if 'extraction_df' not in locals() and 'extraction_df' not in globals():
    file_path = output_filename
    
    # Check if the file exists
    if path.exists(file_path):
        extraction_df = pd.read_csv(file_path)
        print(f"Read `extraction_df` from {file_path}")
    else:
        raise FileNotFoundError(f"File not found: {file_path}")
    
extraction_df

# %% [markdown]
# ### Diagnostics at a glance

# %%
n_phds = len(pubs_df)

# Count the number of non-missing match information values (value exists == PhD confirmed, c.f. dd47028f21f3dfcf3967b54bc8d4a93da2cb4fd2)
n_confirmed_phds = (
    extraction_df.drop_duplicates(subset=['phd_name', 'phd_id'])['phd_match_by']
    .notna()
    .sum()
)

n_considered_sups = pubs_high_contrib_df[contributor_cols].notna().sum().sum()

n_confirmed_sups = extraction_df['contributor_confirmed'].sum()

print(f"Out of {n_phds} PhD candidates, we confirmed {n_confirmed_phds}.")

print(f"Out of {n_considered_sups} considered contributors, we confirmed {n_confirmed_sups} as a supervisor.")

print(f"We managed to find contributors with {extraction_df['n_shared_inst_grad'].sum()} shared institutions and {extraction_df['n_shared_pubs'].sum()} shared publications!")

pubs_high_contrib_df[contributor_cols]

# %% [markdown]
# ### PhDs that we could not find in OpenAlex.

# %%
# Step 1: Filter extraction_df for rows with phd_id = NaN
extraction_none_df = extraction_df.query("phd_id != phd_id")

# Step 2: Filter pubs_unabbrev_df for matching phd_names; then sort and export
pubs_phd_not_confirmed_df = (
    pubs_unabbrev_df
    .query("phd_name in @extraction_none_df.phd_name")
    .sort_values(by=["year", "institution"])   # sort by multiple columns
)

# Export to CSV without the DataFrame index
pubs_phd_not_confirmed_df.to_csv("data/output/phds_not_confirmed.csv", index=False)

pubs_phd_not_confirmed_df

# %%
plotter = PhDMatchPlotter(extraction_df)
ax = plotter.plot()
plt.show()

# %%
plotter = ContributorMatchPlotter(extraction_df)
ax = plotter.plot()
plt.show()


