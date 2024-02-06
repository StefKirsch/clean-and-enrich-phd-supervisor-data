import pandas as pd

def remove_non_person_contributors_and_export(df, csv_path, nlp):
    global removed_contributors  # Use the global list to track removed contributors
    removed_contributors = []  # Ensure the list is empty before starting

    # Inner function to apply on the 'contributor' column
    def is_person_name(name):
        doc = nlp(name)
        is_person = any(ent.label_ == "PER" for ent in doc.ents)
        if not is_person:
            removed_contributors.append(name)
        return is_person

    # Filter the DataFrame based on whether the contributor is a person
    filtered_df = df[df['contributor'].apply(is_person_name)]

    # Export the list of removed contributors to a CSV file
    if removed_contributors:  # Check if there are any removed contributors
        removed_contributors_df = pd.DataFrame(removed_contributors, columns=['non_person_contributors'])
        removed_contributors_df.to_csv(csv_path, header=False, index=False)
    
    return filtered_df