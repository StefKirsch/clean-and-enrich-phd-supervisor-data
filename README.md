# Clean and enrich PhD supervisor data

## Prerequisites

We recommend installing Python via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a minimal setup. This ensures an isolated and controlled Python environment. If you already have Python installed, you can skip this step.

## Setup

1. **Clone or download the repository** to your local machine.

2. **Place [raw data](data/raw/README.md) in `data/raw`** to run the notebook.

3. **Navigate to the project directory** in the terminal or command prompt.

4. **Install the dependencies**  
   You have two options:

### Simple Installation (No Virtual Environment)
   If you do not wish to use a virtual environment, you can install the dependencies directly:
   
   ```bash
   pip install -r requirements.txt
   ```

   This will install all the required packages directly into your current Python environment.

### Recommended Workflow (Using a Virtual Environment)
   It’s recommended to use `venv` to create an isolated Python environment for the project.

   a) **Create a Virtual Environment:**

   Run the following command to create a virtual environment in the project directory:
   
   ```bash
   python -m venv venv
   ```

   This will create a `venv` directory in your project.

   b) **Activate the Virtual Environment:**

   - On Windows:
   
     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:
   
     ```bash
     source venv/bin/activate
     ```

   c) **Install Dependencies:**

   With the virtual environment activated, install the required dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Jupyter Notebook:**

   After setting up the environment and installing dependencies, start the Jupyter notebook:
   
   ```bash
   jupyter notebook
   ```