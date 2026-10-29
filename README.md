# Clean and enrich PhD supervisor data

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). It automatically downloads Python 3.12 if needed and manages the project environment and dependencies.

## Setup

1. **Clone or download the repository** to your local machine.

2. **Place [raw data](data/raw/README.md) in `data/raw`** to run the notebook.

3. **Navigate to the project directory** in the terminal or command prompt.

4. **Install Python and the dependencies:**
   
   ```bash
   uv sync --locked
   ```

   This creates `.venv` using `pyproject.toml` and `uv.lock`. No environment activation is needed.

5. **Run Jupyter Notebook:**

   After setting up the environment and installing dependencies, start the Jupyter notebook:
   
   ```bash
   uv run jupyter notebook
   ```
   
6. **(Optional) Use the polite pool or OpenAlex Premium**

    In order to use the [polite pool](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication#the-polite-pool), create a file `contact_email.txt` in the root folder of the project and specify your e-mail address.

    ```contact_email.txt
    mail@example.com
    ```

7. **Run the cleaning notebook, then the matching script:**
   
   1. Open and run `clean_data.ipynb` in Jupyter.
   2. From the project directory, run `uv run Open_Alex_Match.py`.

   Run other Python scripts with `uv run path/to/script.py`.

## Reducing OpenAlex API costs

The matching workflow reduces API usage through the following choices:

- PhD matching checks the first 10 author-name results and retrieves at most 100 works per candidate, without fetching additional pages. OpenAlex charges per request, so increasing the works page from 25 to 100 results adds no API usage cost ([pricing](https://help.openalex.org/access/example-costs/), [page limits](https://help.openalex.org/api/paging/)).
- Shared-publication and thesis-coauthor checks use the authorships already returned with the selected PhD's works, avoiding separate publication queries for each contributor.
- Candidates must pass the name check before their works are fetched. Contributor searches are skipped when the PhD is unconfirmed or has no affiliation in the target years.
- API responses are cached locally, and an existing pilot-supervisor lookup is loaded from disk to avoid repeating requests.
- Sampling, contributor selection, and gold-standard mode in [dataset_config.yaml](dataset_config.yaml) allow smaller runs before processing the full dataset.

Limiting retrieval reduces requests but can miss matching evidence outside the fetched works and authorships.

### OpenAlex API pricing

As of 29 October 2026, [OpenAlex pricing](https://help.openalex.org/access/example-costs/) charges the following rates in USD per API request (one results page):

| Request type | Cost per request | Use in the extraction |
| --- | ---: | --- |
| Author-name search (`/authors?search=...`) | $0.001 | One search per PhD, examining the first 10 candidates, and one search per considered contributor, examining up to 100 candidates. Contributor searches run only when the PhD is confirmed and has an affiliation in the target years. |
| Filtered works list (`/works?filter=author.id:...`) | $0.0001 | One request per PhD candidate that passes the name check, returning up to 100 works. At most 10 such requests per PhD; no additional pages or contributor works are fetched. |

Single-entity lookups by ID or DOI are free. Fetching an author's works uses the paid filtered-list endpoint above. Thesis-title similarity is computed locally with [SPECTER](https://github.com/allenai/specter). Affiliation and coauthorship checks reuse the returned data without additional API requests.

The current gold-standard sample contains 500 rows, of which 497 have a PhD name and are processed. Assuming two contributors per PhD[^1], a fresh run with no cache hits has this conservative request budget:

[^1]: By default, we retain two contributors per PhD.

| Extraction step | Requests | API usage cost |
| --- | ---: | ---: |
| PhD name searches | 497 | $0.497 |
| Contributor name searches | 994 | $0.994 |
| Works lists for up to 10 candidates per PhD | Up to 4,970 | Up to $0.497 |
| **Total (up to)** | **6,461** | **$1.988** |

As of 29 October 2026, OpenAlex includes $1 of free daily usage with an API key. Spreading extraction across daily allowances can make the out-of-pocket cost $0; the script stops when the daily allowance and prepaid balance are exhausted, and cached responses can be reused on a later run.
