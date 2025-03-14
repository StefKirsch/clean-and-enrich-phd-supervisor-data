import yaml
import pandas as pd

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

def load_dataset(config: dict, dataset_path: str) -> pd.DataFrame:
    """
    Loads and subsets a dataset according to the provided configuration.

    Args:
        config (dict): Dictionary with keys:
            - nrows (int or None)
            - use_dataset (str)
            - random_sample (bool)
            - batch (int or None)
        dataset_path (str): Path to the CSV dataset.
    
    Returns:
        pd.DataFrame: Subset of data according to config settings.
    """
    # Read full dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Total rows: {len(df)}")

    domain = config.get('domain', 'all')
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
        # Note: 'seed' has been replaced by 'random_state'
        df = df.sample(n=min([nrows, len(df)]), random_state=42)
        print(f"Sampled rows: {len(df)}")
    else:
        print("Taking the full (sub)set...")

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
