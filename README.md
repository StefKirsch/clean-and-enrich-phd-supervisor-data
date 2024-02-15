# Clean and enrich PhD supervisor data

## Prerequisites

[Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Setup

1. **Clone or download the repository** to your local machine.

2. **Place raw data in `data/raw`** to run the notebook.

3. **Navigate to the project directory** in the terminal or command prompt.

4. **Set up the environment**

You can do this via the Anaconda navigator or via the Anaconda Promot Terminal

### Using Anaconda Navigator

a) **Import the Environment:**

- Open Anaconda Navigator.
- Go to the "Environments" tab.
- Click on "Import" at the bottom.
- In the Import dialog, click the folder icon and select the `enrich-phd-sup-data-conda-env.yaml` file from this repository.
- Click "Import".

b) **Activate the Environment:**

- Once the environment has been created, you can activate it by selecting it in the "Environments" tab.
- To launch tools like Jupyter notebooks, switch to the "Home" tab, select the environment from the dropdown menu, and click the "Launch" button next to the tool you want to use.

### Using the Console

a) **Create the Environment:**
   Open a terminal or command prompt and navigate to the directory containing this project. Then, run the following command:

   ```sh
   conda env create -f enrich-phd-sup-data-conda-env.yaml
   ```

b) **Activate the Environment:**
   To activate the newly created environment, use:

   ```sh
   conda activate enrich-phd-sup-data-conda-env
   ```

c) **Run Jupyter Notebook** by typing

   ```sh
   jupyter notebook
   ```
