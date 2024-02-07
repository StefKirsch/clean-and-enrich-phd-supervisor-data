import pandas as pd
from nameparser import HumanName

def remove_non_person_contributors_and_export(df, csv_path, nlp, whitelist=[], blacklist=[]):
    global removed_contributors
    removed_contributors = []  # Reset the list for each call

    def is_person_name(name):
        # Short-circuit detection with whitelist and blacklists
        if name in whitelist: 
            return True
        if name in blacklist:
            return False
        doc = nlp(name)
        is_person = any(ent.label_ == "PER" for ent in doc.ents)
        if not is_person:
            removed_contributors.append(name)
        return is_person

    filtered_df = df[df['contributor'].apply(is_person_name)]

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