import ipywidgets as widgets
from IPython.display import display

# Define dataset presets
CONFIGS = {
    "quick_test": {"NROWS": 25, "use_dataset": "biomedical_5y"},
    "biomedical_subset": {"NROWS": 2573, "use_dataset": "biomedical_5y"},
    "full_dataset": {"NROWS": None, "use_dataset": None},
}

class DatasetSelector:
    def __init__(self, default="quick_test"):
        self.config_dropdown = widgets.Dropdown(
            options=CONFIGS.keys(),
            value=default,
            description="Preset:",
            style={'description_width': 'initial'}
        )
        
        self.output = widgets.Output()
        self.config_dropdown.observe(self.update_config, names='value')
        self.NROWS = None
        self.use_dataset = None
        self.output_filename = None
        
        # Initialize with default selection
        self.update_config({'new': self.config_dropdown.value})

    def update_config(self, change):
        selected = change['new']
        config = CONFIGS[selected]
        
        self.NROWS = config["NROWS"]
        self.use_dataset = config["use_dataset"]
        self.output_filename = (
            "data/output/matched_pairs_full.csv"
            if self.NROWS is None
            else f"data/output/matched_pairs_{self.NROWS}_rows.csv"
        )
        
        with self.output:
            self.output.clear_output()
            print(f"Running analysis with preset: {selected}")
            print(f"NROWS: {self.NROWS}, use_dataset: {self.use_dataset}")
            print(f"Output file: {self.output_filename}")

    def display(self):
        display(self.config_dropdown, self.output)

# Function to create and return an instance of the selector
def get_dataset_selector(default="quick_test"):
    return DatasetSelector(default)
