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


