import ipywidgets as widgets
from IPython.display import display

# Define dataset presets
CONFIGS = {
    "quick test": {"NROWS": 10, "use_dataset": None},
    "200 rows": {"NROWS": 200, "use_dataset": None},
    "biomedical subset": {"NROWS": 2573, "use_dataset": "biomedical_5y"},
    "random subset": {"NROWS": 2573, "use_dataset": None},
    "full dataset": {"NROWS": None, "use_dataset": None},
}

class DatasetSelector:
    def __init__(self, default="quick test"):
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

    def display(self):
        display(self.config_dropdown, self.output)

# Function to create and return an instance of the selector
def get_dataset_selector(default="quick test"):
    return DatasetSelector(default)
