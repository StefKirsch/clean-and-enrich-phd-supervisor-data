import yaml
import pandas as pd
from typing import Optional

def read_config(config_path: str) -> dict:
    """
    Reads a YAML configuration file.

    Args:
        config_path (str): Path to the config file.
    
    Returns:
        dict: Parsed configuration.
    """
    print(f"Reading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_dataset(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Loads (and subsets) the dataset according to the provided configuration.

    Args:
        config (dict): Dictionary with keys:
            - dataset_path (str)
            - dataset_path_sample_gold_standard (str)
            - output_filename (str)
            - use_sample_for_gold_standard (bool)
            - domain (str or None)
            - chunk (int or None)
            - nrows (int or None)
            - min_rank_contrib (int or None)
        
    Returns:
        pd.DataFrame: (Sub)set of data according to config settings.
    """
    
    # If no config is provided, use an empty dict, which will resort to the full dataset
    if config is None:
        config = {}
    
    use_sample_for_gold_standard = config.get('use_sample_for_gold_standard', None)
    
    if use_sample_for_gold_standard:
        actual_dataset_path = config.get('dataset_path_sample_gold_standard', None)
        print(f"Using the data subset for the manual gold standard.")
    else:
        actual_dataset_path = config.get('dataset_path', None)
        print(f"Dataset loaded.")
    
    # Read the respective dataset
    df = pd.read_csv(actual_dataset_path)
        
    print(f"Total rows: {len(df)}")

    # We skip further sampling if we are using the sample for the manual gold standard
    if not use_sample_for_gold_standard:
        domain = config.get('domain', None)
        chunk = config.get('chunk', None)
        nrows = config.get('nrows', None)
        print(f"Using domain={domain}, chunk={chunk}, nrows={nrows}")
        
        # Subset the dataset based on 'domain'
        if domain:
            print(f"Subsetting data for domain={domain}...")
            df = df[df["domain"] == domain]
            print(f"Subsetted rows: {len(df)}")
        else:
            print("No subsetting per domain...")

        # Subset the dataset based on 'batch'
        if chunk is not None and nrows is not None: # we have a batch and a row number
            print(f"Splitting DataFrame into chunks, then selecting batch #{chunk}...")
            chunks = list(chunk_dataframe(df, chunk_size=nrows))
            idx = min([chunk, len(chunks) - 1])
            df = chunks[idx]
            print(f"Selected chunk index: {idx}. Rows in this chunk: {len(df)}")
        elif nrows is not None: # we have a row number
            print(f"Randomly sampling up to {nrows} rows...")
            df = df.sample(n=min([nrows, len(df)]), random_state=42)
            print(f"Sampled rows: {len(df)}")
        else:
            print("Taking the full (sub)set...")
    
    # Drop rows that don't have a PhD name. 
    # These can occur in the gold standard subset
    df = df.dropna(subset=["phd_name"])
    
    return df

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """
    Splits a DataFrame into chunks of specified size.
    
    Args:
        df (pd.DataFrame): The DataFrame to chunk.
        chunk_size (int): Number of rows per chunk.
    
    Yields:
        pd.DataFrame: A chunk of the original DataFrame.
    """
    nrows = len(df)
    for start in range(0, nrows, chunk_size):
        end = start + chunk_size
        yield df.iloc[start:end]
