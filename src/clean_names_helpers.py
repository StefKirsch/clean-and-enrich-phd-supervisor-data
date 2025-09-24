import pandas as pd
from nameparser import HumanName
from tqdm.notebook import tqdm  # Import tqdm for Jupyter Notebook
import spacy
from spacy.cli import download
import unicodedata
import re

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


def format_name_to_lastname_firstname(name):
    human_name = HumanName(name)
    
    # Extract the last name and first name + middle name
    last_name = human_name.last
    first_names = human_name.first + ' ' + human_name.middle

    # Combine last name and first names, giving only last name if first names are missing
    formatted_name = f"{last_name}, {first_names}" if first_names.strip() else last_name

    return formatted_name.strip()


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


def merge_near_duplicates_on_col(df: pd.DataFrame, merge_col: str = "institution") -> pd.DataFrame:
    """
    Handle duplicate entries that only differ in one column by merging them together, producing a
    set of the unique values in that column per duplicate group. 
    
    # NOTE
    # This is currently unused, as it can be very difficult identify functionally duplicate columns at this stage of the pipeline.
    # c.f. #46
    """
    other_cols = [c for c in df.columns if c != merge_col]

    # Combine all unique values into a tuple of values, preserving all version of merge_col we came across in the duplicates 
    def merge_vals(s: pd.Series):
        vals = pd.unique(s.dropna())
        return vals[0] if len(vals) == 1 else tuple(vals)
    
    merged = (
        df
        .groupby(other_cols, as_index=False, dropna=False, sort=False)[merge_col]
        .agg(merge_vals)
    )
    
    print(f"Merged {len(df)-len(merged)} duplicates that only differ in the '{merge_col}' column.")
    
    return merged

# All dash-like characters we want to treat as "hyphen between surnames"
_DASH_CHARS = "-‐-‒–—―−"  # hyphen-minus, hyphen, non-breaking hyphen, figure, en, em, horiz bar, minus
_DASH_SPLIT_RE = re.compile(rf"\s*[{re.escape(_DASH_CHARS)}]\s*")

def _first_alpha(s: str):
    for ch in unicodedata.normalize("NFKD", s):
        if ch.isalpha():
            return ch.upper()
    return None

def _letters_upper(s: str) -> str:
    # accent-insensitive, keep only letters, uppercase
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if ch.isalpha()).upper()

def first_token_of_given(hn: HumanName) -> str:
    given = " ".join(p for p in [hn.first, hn.middle] if p).strip()
    return given.split()[0] if given else ""

def _surname_components(hn: HumanName):
    """
    Return (components, has_dash):
      - If surname contains a dash-like char, split on it and return both parts (normalized).
      - Otherwise, return just the *last word* of the surname (normalized).
    """
    last = (hn.last or "").strip()
    if not last:
        return [], False

    has_dash = any(ch in _DASH_CHARS for ch in last)
    if has_dash:
        # Split on any dash-like char, allowing spaces around it (e.g., "Wagner - Cremer")
        raw_parts = [p for p in _DASH_SPLIT_RE.split(last) if p.strip()]
        parts = [_letters_upper(p) for p in raw_parts if _letters_upper(p)]
        return parts, True
    else:
        last_word = last.split()[-1]
        return ([_letters_upper(last_word)] if _letters_upper(last_word) else []), False

def surname_word_match(name_a: str, name_b: str) -> bool:
    """
    Compare surnames with this rule:
      - If *either* surname is hyphenated (any dash-like char), match if *any* hyphen component equals
        a component of the other surname (the other contributes either its own hyphen parts or just its last word).
      - If neither is hyphenated, match only on the *last word*.
    Comparison is accent- and punctuation-insensitive.
    """
    hn_a, hn_b = HumanName(name_a), HumanName(name_b)
    comps_a, a_hyph = _surname_components(hn_a)
    comps_b, b_hyph = _surname_components(hn_b)
    if not comps_a or not comps_b:
        return False
    if a_hyph or b_hyph:
        return bool(set(comps_a) & set(comps_b))
    else:
        # both non-hyphenated: compare last words only
        return comps_a[0] == comps_b[0]

def first_given_initial_match(name_a: str, name_b: str) -> bool:
    hn_a = HumanName(name_a)
    hn_b = HumanName(name_b)
    a_init = _first_alpha(first_token_of_given(hn_a))
    b_init = _first_alpha(first_token_of_given(hn_b))
    return a_init is not None and b_init is not None and a_init == b_init

def name_sanity_check(name_a: str, name_b: str) -> bool:
    """
    Sanity check if name_a and name_b could realistically refer to the same person.
    True if the first given-name initial is the same AND the last surname word is the same
    (both checks accent/punctuation-insensitive).
    """
    return first_given_initial_match(name_a, name_b) and surname_word_match(name_a, name_b)