import requests
import pandas as pd
from io import BytesIO
import re
from string import punctuation
from num2words import num2words

def fetch_supervisors_from_pilot_dataset(repo_url, file_extension=".xlsx", verbosity=False):
    """
    Fetches all files with a specified extension from a GitHub directory,
    downloads them using the correct raw URL format, and combines them into a single pandas DataFrame.

    Parameters:
        repo_url (str): The URL of the GitHub directory containing the files.
        branch (str): The GitHub branch reference (default: "refs/heads/main").
        file_extension (str): The extension of the files to download (default: ".xlsx").
        verbosity (bool): If True, prints progress and errors (default: False).

    Returns:
        pd.DataFrame: A combined DataFrame containing data from all downloaded files.
    """
    # Fetch the HTML content of the directory
    try:
        response = requests.get(repo_url)
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to fetch directory listing from {repo_url}: {e}")

    # Extract file names ending with the specified extension
    try:
        file_names = re.findall(r'href=".*?/([\w\-]+%s)"' % re.escape(file_extension), response.text)
        file_names = list(set(file_names)) # remove duplicates
        if verbosity:
            print(f"Found {len(file_names)} files with extension '{file_extension}'.")
    except Exception as e:
        raise ValueError(f"Failed to parse file names from directory listing: {e}")
    
    # Base URL for raw file downloads
    base_url = repo_url.replace("/tree/", "/raw/refs/heads/")

    # Generate full URLs for the files
    file_urls = [f"{base_url}/{file_name}" for file_name in file_names]

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through the file URLs and process them
    for file_url in file_urls:
        try:
            # Download the file
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            
            # Read the file into a DataFrame
            df = pd.read_excel(BytesIO(file_response.content))
            
            # Append the DataFrame to the list
            dataframes.append(df)
            
            if verbosity:
                print(f"Successfully processed file: {file_url}")
        except Exception as e:
            if verbosity:
                print(f"Error processing file {file_url}: {e}")

    # Combine all DataFrames into a single DataFrame
    if not dataframes:
        raise ValueError("No valid files were processed. Combined DataFrame is empty.")
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Ensure the 'supervisor' column exists and extract unique values
    if 'supervisor' not in combined_df.columns:
        raise ValueError("The 'supervisor' column does not exist in the combined dataset.")

    unique_supervisors = combined_df.query("supervisor.notnull()")['supervisor'].unique().tolist()

    if verbosity:
        print(f"Found {len(unique_supervisors)} unique supervisors.")

    return unique_supervisors


def remove_illegal_title_characters(text: str) -> str:
    # Remove punctuation and pipe characters
    # Punctuation is removed with the intention to make the matching by literal title more robust. 
    # OpenAlex's search() parameter is pretty robust against punctuation, so we technically don't need it, but it also does not hurt. 
    # Pipe characters must be removed on the other hand, because they are misinterpreted as OR by pyalex, 
    # which can cause unexpected behavior and is also not supported by the search() parameter.
    punctuation_pattern = '[' + re.escape(punctuation) + ']'
    cleaned_text = re.sub(punctuation_pattern, '', text)
    return cleaned_text


def ordinal(n):
    return num2words(n, to='ordinal_num')