# This script can be run to verify the un-abbreviation when translating 
# institution abbreviations to a name that can be found in OpenAlex

from pyalex import Institutions

# Dictionary to map institution abbreviations to full display names
# The abreviation list were obtained from all institutions that are named in the NARCIS dataset
# All long names were discovered with a bit of detective work and then looked up in OpenAlex and a verbatim version that is named there was added. 
# The version that is chosen is either a display name or a display name alternative verbatim from Open alex
institution_translation = {
    'amcpub': 'Amsterdam UMC Location University of Amsterdam', # UMC location AMC, historically associated with UvA, https://api.openalex.org/institutions?search=vumc
    'buas': 'Breda University of Applied Sciences', # https://api.openalex.org/institutions?search=Breda%20University%20of%20Applied%20Sciences
    'cwi': 'Centrum Wiskunde & Informatica', # https://api.openalex.org/institutions?search=Centrum%20Wiskunde%20Informatica
    'eur': 'Erasmus University Rotterdam', # https://api.openalex.org/institutions?search=Naturalis%20Biodiversity%20Center
    'amsterdam_pure': 'Amsterdam University of Applied Sciences', # https://api.openalex.org/institutions?search=Amsterdam%20University%20of%20Applied%20Sciences
    'hanzepure': 'Hanze University of Applied Sciences', # https://api.openalex.org/institutions?search=Hanze%20University%20of%20Applied%20Sciences
    'lumc': 'Leiden University Medical Center', # https://api.openalex.org/institutions?search=Leiden%20University%20Medical%20Center
    'naturalis': 'Naturalis Biodiversity Center', # https://api.openalex.org/institutions?search=Naturalis%20Biodiversity%20Center
    'ou': 'Open University of the Netherlands', # https://api.openalex.org/institutions?search=Open%20Universiteit%20Nederland
    'ru': 'Radboud University Nijmegen', # https://api.openalex.org/institutions?search=Radboud%20University
    'rug': 'University of Groningen', # https://api.openalex.org/institutions?search=University%20of%20Groningen
    'tno': 'Netherlands Organisation for Applied Scientific Research', # https://api.openalex.org/institutions?search=Netherlands%20Organisation%20for%20Applied%20Scientific%20Research
    'tud': 'Delft University of Technology', # https://api.openalex.org/institutions?search=Delft%20University%20of%20Technology
    'tue': 'Eindhoven University of Technology', # https://api.openalex.org/institutions?search=Eindhoven%20University%20of%20Technology
    'ul': 'Leiden University', # https://api.openalex.org/institutions?search=Universiteit%20Leiden
    'uls': 'Utrecht University', # this abbreviation ocurred only for one thesis. I looked it up and it was written with UU.
    'um': 'Maastricht University', # https://api.openalex.org/institutions?search=Universiteit%20Maastricht
    'umcu': 'Maastricht University Medical Centre', # https://api.openalex.org/institutions?search=Maastricht%20University%20Medical%20Centre
    'ut': 'University of Twente', # https://api.openalex.org/institutions?search=Universiteit%20Twente 
    'uu': 'Utrecht University',
    'uvapub': 'University of Amsterdam',
    'uvh': 'University of Humanistic Studies',
    'uvt': 'Tilburg University',
    'vu': 'Vrije Universiteit Amsterdam', # https://api.openalex.org/institutions?search=Vrije%20Universiteit%20Amsterdam
    'vumc': 'Amsterdam UMC Location Vrije Universiteit Amsterdam', # https://api.openalex.org/institutions?search=Amsterdam%20UMC%20Location%20VUmc
    'wur': 'Wageningen University & Research' # https://api.openalex.org/institutions?search=Wageningen%20University%20Research
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
