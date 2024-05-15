import pandas as pd
from nameparser import HumanName
from tqdm.notebook import tqdm  # Import tqdm for Jupyter Notebook
import spacy
from spacy.cli import download

def remove_non_person_contributors_and_export(df, csv_path, nlp, whitelist=[], blacklist=[]):
    global removed_contributors
    removed_contributors = []  # Reset the list for each call

    def is_person_name(name):
        # Short-circuit detection with whitelist and blacklists
        if name in whitelist: 
            return True
        if name in blacklist:
            return False
        doc = nlp(name) # process with model
        is_person = any(ent.label_ == "PER" for ent in doc.ents) # check if any named entitties is a person
        if not is_person:
            removed_contributors.append(name) # add removed entry to log
        return is_person

    # Wrap df['contributor'].apply in tqdm
    tqdm.pandas(desc="Removing invalid contributors")
    filtered_df = df[df['contributor'].progress_apply(is_person_name)]

    if removed_contributors:  # Only export if there are names to export
        removed_contributors_df = pd.DataFrame(removed_contributors, columns=['non_person_contributors'])
        removed_contributors_df.to_csv(csv_path, header=False, index=False)

    return filtered_df

def format_name_to_lastname_initials(name):
    human_name = HumanName(name)
    # Extract the last name
    last_name = human_name.last
    # Extract the first name and/or initials
    first_names = human_name.first + ' ' + human_name.middle
    # Convert first names to initials, keeping existing initials as is
    initials = ''.join([f"{name[0]}." if name else '' for name in first_names.split()])
    # Combine last name and initials
    formatted_name = f"{last_name}, {initials}" if initials else last_name
    return formatted_name.strip()


def format_initials(initials):
    # Split the string into a list of initials
    initials_list = initials.split()
    # Format each initial with a period
    formatted_initials = '.'.join(initials_list) + '.'
    return formatted_initials


def ensure_and_load_spacy_model(model_name):
    """
    Ensures that the specified spaCy model is downloaded and loaded.
    
    Parameters:
    model_name (str): The name of the spaCy model to check and download if necessary.
    
    Returns:
    nlp (Language): The spaCy Language object for the specified model.
    """
    try:
        # Try loading the model
        nlp = spacy.load(model_name)
        print(f"{model_name} is already installed.")
    except OSError:
        # If the model is not found, download it
        print(f"{model_name} not found, downloading...")
        download(model_name)
        # Load the model after downloading
        nlp = spacy.load(model_name)
        print(f"{model_name} has been successfully downloaded.")
    print(f"{model_name} has been loaded!")
    return nlp
