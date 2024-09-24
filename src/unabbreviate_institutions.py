# This script can be run to verify the un-abbreviation when transalting 
# institution abbreviatrons to a name that can be found in OpenAlex

from pyalex import Institutions

# Dictionary to map institution abbreviations to full display names
institution_translation = {
    'amcpub': 'Amsterdam UMC',
    'buas': 'Breda University of Applied Sciences',
    'cwi': 'Centrum Wiskunde & Informatica',
    'eur': 'Erasmus University Rotterdam',
    'amsterdam_pure': 'University of Amsterdam',
    'hanzepure': 'Hanze University of Applied Sciences',
    'lumc': 'Leiden University Medical Center',
    'naturalis': 'Naturalis Biodiversity Center',
    'ou': 'Open Universiteit',
    'ru': 'Radboud University',
    'rug': 'University of Groningen',
    'tno': 'Netherlands Organisation for Applied Scientific Research',
    'tud': 'Delft University of Technology',
    'tue': 'Eindhoven University of Technology',
    'ul': 'Leiden University',
    'uls': '', # I wasn't able to find out what this means
    'um': 'Maastricht University',
    'umcu': 'University Medical Center Utrecht',
    'ut': 'University of Twente',
    'uu': 'Utrecht University',
    'uvapub': 'University of Amsterdam',
    'uvh': 'University of Humanistic Studies',
    'uvt': 'Tilburg University',
    'vu': 'Vrije Universiteit Amsterdam',
    'vumc': 'Amsterdam UMC, VUmc location',
    'wur': 'Wageningen University & Research'
}

# Function to un-abbreviate the 'institution' column in a DataFrame
def unabbreviate_institutions(df, column):
    """
    This function replaces institution abbreviations in the specified column of a DataFrame
    with their full institution names using the institution_translation dictionary.
    
    Parameters:
    - df: pandas DataFrame that contains the institution column
    - column: the name of the column containing institution abbreviations (as a string)
    
    Returns:
    - A pandas DataFrame with the updated column where abbreviations are replaced with full names
    """
    df[column] = df[column].map(institution_translation).fillna(df[column])
    return df

def check_institutions_openalex(institution_dict):
    not_found = []
    
    for abbr, display_name in institution_dict.items():
        try:
            # Search for the institution in OpenAlex using the display name
            result = Institutions().search(display_name).get()
            
            # Check if any result is returned
            if result:
                print(f"Institution '{display_name}' found in OpenAlex.")
            else:
                print(f"Institution '{display_name}' not found in OpenAlex.")
                not_found.append(display_name)
        
        except Exception as e:
            print(f"Error searching for '{display_name}': {e}")
            not_found.append(display_name)
    
    # Return a list of institutions that were not found
    return not_found


if __name__ == '__main__':
    not_found_institutions = check_institutions_openalex(institution_translation)

    # Print any institutions that were not found
    if not_found_institutions:
        print(f"These institutions were not found in OpenAlex: {not_found_institutions}")
    else:
        print("All institutions were found in OpenAlex.")
