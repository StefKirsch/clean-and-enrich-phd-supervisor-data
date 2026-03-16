import pandas as pd
from nameparser import HumanName
import spacy
from spacy.cli import download
from tqdm.notebook import tqdm  # Import tqdm for Jupyter Notebook
import unicodedata
import re
from pathlib import Path

ORG_WORDS = {
    # English: institutions / units
    "academy", "academies", "center", "centers", "centre", "centres", "department", "departments", "division",
    "divisions", "faculty", "faculties", "group", "groups", "institute", "institutes", "lab", "labs",
    "laboratory", "laboratories", "office", "offices", "programme", "programmes", "program", "programs",
    "research", "school", "schools", "section", "sections", "study", "studies", "unit", "units", "university",
    "universities",

    # English: subject / field words
    "accounting", "algorithm", "algorithms", "algebra", "analysis", "audition", "business", "care", "careers",
    "child", "clinical", "cognition", "complex", "databases", "democratic", "dentistry", "development",
    "diseases", "education", "emotion", "energy", "engineering", "environmental", "experimental", "finance",
    "governance", "health", "history", "human", "humanism", "immunity", "infections", "informatics", "language",
    "lasers", "law", "learning", "management", "marketing", "mathematics", "matter", "medicine", "metabolism",
    "networks", "nursing", "orthodontics", "perception", "pharmacy", "philosophy", "physics", "planning",
    "policy", "praxis", "probability", "public", "security", "social", "stochastics", "storytelling",
    "tectonics", "theory", "unknown", "xenon",

    # Dutch: institutions / units
    "academie", "academies", "afdeling", "afdelingen", "bureau", "bureaus", "centrum", "centra", "divisie",
    "divisies", "eenheid", "eenheden", "faculteit", "faculteiten", "groep", "groepen", "instituut", "instituten",
    "laboratorium", "laboratoria", "onderzoek", "onderzoeksgroep", "onderzoeksgroepen", "programma", "programma's",
    "sectie", "secties", "studie", "studies", "universiteit", "universiteiten",

    # Dutch: subject / field words
    "beeldvorming", "beleid", "farmacie", "geschiedenis", "informatica", "klinisch", "medisch", "onderwijs",
    "orthodontie", "overig", "overige", "publiek", "zorg", "identiteit",

    # Acronyms / special
    "ACTA", "AGCI", "AIMMS", "AMIBM", "ANTARES", "ARCNL", "CBITE", "CLUE", "CTC", "CTR", "CvE", "EMGO",
    "ESoE", "Eurandom", "FMG", "GRAPPA", "IBBA", "IBIS", "ICIS", "IHEF", "IHS", "INTERVICT", "IOO", "IViR",
    "KNO", "LEARN", "Leiden", "MRI", "NUTRIM", "Sixma", "TILT", "TNO", "WZI", "nvt",
}

ORG_WORD_PREFIXES = (
    "rs:", "mumc+:", "cca -", "nca -", "ams -", "api ",
    "department ", "faculty ", "section ", "research ", "academic ",
    "school ", "center ", "centre ", "institute ", "university "
)

ORG_WORD_ENDINGS = (
    # English: broad academic / discipline endings
    "atics", "ation", "chemistry", "circulation", "energy", "health", "iatrics", "iatry", "ineering",
    "informatics", "istic", "istics", "lab", "literature", "lysis", "magnetic", "magnetics", "media", "medica",
    "mechanics", "metrics", "metry", "molecular", "nomics", "ologies", "ology", "onomies", "onomy", "physics",
    "science", "sciences", "spectroscopy", "stics", "surgery", "system", "systems", "therapy", "vision",

    # Dutch: broad academic / discipline endings
    "atie", "bibliotheek", "chemie", "chirurgie", "informatica", "iatrie", "iatrieën", "istiek", "kunde",
    "logie", "logieën", "metrie", "nomie", "nomieën", "pedie", "recht", "studie", "studies", "techniek",
    "therapie", "wetenschap", "wetenschappen",
)

ORG_RE = re.compile(
    r"(?:\b(?:and)\b|\b(?:"
    + "|".join(re.escape(w) for w in sorted(ORG_WORDS, key=len, reverse=True))
    + r")\b|\w*(?:"
    + "|".join(re.escape(f) for f in ORG_WORD_ENDINGS)
    + r")\b)",
    re.I
)

# Rules for person parsing
NAME_TOKEN = r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'`’.-]+"
PARTICLE = r"(?:van|von|de|del|der|den|ten|ter|te|la|le|du|di|da|dos|des|el|al|bin|ibn)"
SURNAME = rf"(?:{PARTICLE}\s+)*{NAME_TOKEN}(?:[-\s](?:{PARTICLE}\s+)?{NAME_TOKEN})*"
BASE_SURNAME = rf"{NAME_TOKEN}(?:[-\s]{NAME_TOKEN})*"
INITIALS = r"(?:[A-Z]\.){1,8}[A-Z]?|[A-Z]{1,6}"
GIVEN = rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN})*"

TITLE_WORD = r"(?:emerit(?:us|a)(?:\s+prof(?:essor)?)?|prof(?:essor)?|drs?|ir|mr|ing|ba|bsc|ma|msc|ph\.?d|hoogleraar)"
TITLES = rf"(?:{TITLE_WORD}\.?\s*,?\s*)*"

ANNOTATED_SURNAME = rf"{SURNAME}(?:\s+\(\s*{TITLE_WORD}\s*\))?"
ANNOTATED_BASE_SURNAME = rf"{BASE_SURNAME}(?:\s+\(\s*{TITLE_WORD}\s*\))?"
PARTICLE_SEQ = rf"(?:{PARTICLE})(?:\s+{PARTICLE})*"

PERSON_PATTERNS = [
    # van der Heide, Tjisse / van Dijk, R.A. / Klucharev, VA (Vasily)
    # Verhoef (Emeritus), Wouter / Preckel, Prof. dr. , Benedikt
    re.compile(
        rf"^{ANNOTATED_SURNAME},\s*(?:{TITLES})?(?:{INITIALS}|{GIVEN})(?:\s+\([^)]+\))?,?$",
        re.I
    ),

    # Gier, de, Han / Son, van, Willem / Velden, van der, Joep
    re.compile(
        rf"^{ANNOTATED_BASE_SURNAME},\s*{PARTICLE_SEQ},\s*(?:{TITLES})?(?:{INITIALS}|{GIVEN})(?:\s+\([^)]+\))?,?$",
        re.I
    ),

    # Wit, G.A. de / Velden, A.A.E.M. van der / Have, K. ten
    re.compile(
        rf"^{ANNOTATED_BASE_SURNAME},\s*(?:{TITLES})?(?:{INITIALS}|{GIVEN})(?:\s+\([^)]+\))?\s+{PARTICLE_SEQ},?$",
        re.I
    ),

    # P.C. Struik / A. van Paassen / S.E.A.T.M. van der Zee / Prof. dr. J. van Dijk
    re.compile(
        rf"^(?:{TITLES})?{INITIALS}\s+{SURNAME},?$",
        re.I
    ),

    # Vivianne Vleeshouwers / Nel Wognum / Prof. Wouter Verhoef
    re.compile(
        rf"^(?:{TITLES})?{GIVEN}\s+{SURNAME},?$",
        re.I
    ),
]

def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\[\s*No Value\s*\]", "", s, flags=re.I)
    s = s.strip().strip('"').strip("'")
    
    # Remove the exact string "Supervisor:"
    s = s.replace("Supervisor:", "")
    
    # Remove publication footnote markers / daggers / stars
    s = re.sub(r"[†‡*⁎⁑§¶‖¤※]+", "", s)

    # Map spacing accent symbols to combining marks
    s = s.translate(str.maketrans({
        '"': "\u0308",   # diaeresis, e.g. honkim"aki -> honkimäki
        "¨": "\u0308",   # diaeresis
        "´": "\u0301",   # acute
        "`": "\u0300",   # grave
        "ˆ": "\u0302",   # circumflex
        "˜": "\u0303",   # tilde
        "°": "\u030A",   # ring above, e.g. Sj°astad -> Sjåstad
        "¸": "\u0327",   # cedilla
    }))

    # Fix misplaced combining marks, e.g. M ́enard -> Ménard / Rottsch¨afer -> Rottschäfer
    s = re.sub(r"\s*([\u0300-\u036f]+)\s*([A-Za-zÀ-ÖØ-öø-ÿĀ-ž])", r"\2\1", s)
    s = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿĀ-ž])\s+([\u0300-\u036f]+)", r"\1\2", s)
    s = unicodedata.normalize("NFC", s)

    s = s.replace("–", "-").replace("—", "-").replace("‐", "-")
    s = re.sub(r"\.\.+", ".", s)           # P.C.. -> P.C.
    s = re.sub(r"\s+", " ", s)

    # Normalize comma junk
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(?:,\s*){2,}", ", ", s)

    return s.strip(" ,")

def rule_classify(s: str):
    """
    Returns:
        decision: 'accept' | 'reject' | 'review'
        reason: short string
    """
    s = normalize_name(s)
    s_lower = s.lower()

    if not s or s == "--":
        return "reject", "empty_or_dash"

    # Explicit strong non-person cues
    if s_lower.startswith(ORG_WORD_PREFIXES):
        return "reject", "non_person_prefix"

    if ORG_RE.search(s):
        return "reject", "org_keyword"

    # Digits usually indicate departments / units / codes in your data
    if any(ch.isdigit() for ch in s):
        return "reject", "contains_digit"

    # Common organization formatting
    if s.count(":") + s.count("/") + s.count("&") + s.count("+") >= 1 and len(s.split()) >= 3:
        return "reject", "org_punctuation_pattern"

    # Very acronym-heavy entries with parentheses are usually non-person
    if re.search(r"\b[A-Z]{2,}\b", s) and "(" in s and ")" in s and "," not in s:
        return "reject", "acronym_parenthesis_pattern"
    
    # Acronym groups like ITC-EOS
    if re.search(r"\b[A-Z]{2,}(?:-[A-Z]{2,})+\b", s):
        return "reject", "acronym_dash_pattern"

    if not any(ch.isalpha() for ch in s):
        return "reject", "no_letters"
    
    # Strong person cues
    for pattern in PERSON_PATTERNS:
        if pattern.match(s):
            return "accept", "person_pattern"

    # Single-token or malformed short strings are ambiguous
    return "review", "ambiguous"

def format_name_to_lastname_firstname(name):
    human_name = HumanName(name)
    
    # Extract the last name and first name + middle name
    last_name = human_name.last
    first_names = human_name.first + ' ' + human_name.middle

    # Combine last name and first names, giving only last name if first names are missing
    formatted_name = f"{last_name}, {first_names}" if first_names.strip() else last_name

    return formatted_name.strip()


def classify_contributors(
    df,
    contributor_col="contributor",
    whitelist=None,
    blacklist=None,
    accept_ambiguous = False,
    reject_csv_path=None,
    review_csv_path=None,
):
    """
    Classify contributors into accept / reject / review.

    Returns:
        accepted_df: rows kept as valid person contributors
        rejected_df: unique rejected contributor strings with counts + reasons
        review_df: unique review contributor strings with counts + reasons
        classified_df: full original df with extra classification columns
    """
    whitelist = whitelist or []
    blacklist = blacklist or []

    # Normalize whitelist/blacklist so matching is robust
    person_lookup = {normalize_name(x) for x in whitelist}
    non_person_lookup = {normalize_name(x) for x in blacklist}

    classified_df = df.copy()
    classified_df[contributor_col] = classified_df[contributor_col].astype(str)
    classified_df["contributor_normalized"] = classified_df[contributor_col].map(normalize_name)

    unique_names = classified_df["contributor_normalized"].dropna().unique()

    decision_map = {}
    reason_map = {}

    for name in tqdm(unique_names, desc="Classifying contributors"):
        # 1) Explicit lists first
        if name in person_lookup:
            decision_map[name] = "accept"
            reason_map[name] = "whitelist"
            continue

        if name in non_person_lookup:
            decision_map[name] = "reject"
            reason_map[name] = "blacklist"
            continue

        # 2) Rules
        decision, reason = rule_classify(name)
        
        # We can choose to accept ambiguous cases or have them go into the review bucket
        if accept_ambiguous and decision=="review":
            decision_map[name] = "accept"
        else:
            decision_map[name] = decision
            
        reason_map[name] = reason
            
    # 3) Map back to full dataframe
    classified_df["contributor_decision"] = classified_df["contributor_normalized"].map(decision_map)
    classified_df["contributor_reason"] = classified_df["contributor_normalized"].map(reason_map)

    # 4) Outputs
    accepted_df = classified_df[classified_df["contributor_decision"] == "accept"].copy().drop(
        columns=["contributor_normalized", "contributor_decision", "contributor_reason"]
    )

    summary_df = (
        classified_df.groupby(
            [contributor_col, "contributor_normalized", "contributor_decision", "contributor_reason"],
            dropna=False
        )
        .size()
        .reset_index(name="n_rows")
        .sort_values(["contributor_decision", "n_rows", contributor_col], ascending=[True, False, True])
    )

    rejected_df = summary_df[summary_df["contributor_decision"] == "reject"].copy()
    
    # We use the contributor_reason here to allow a review bucket even if we accept ambiguous cases
    review_df = summary_df[summary_df["contributor_reason"] == "ambiguous"].copy()

    # 5) Write logs
    if reject_csv_path:
        Path(reject_csv_path).parent.mkdir(parents=True, exist_ok=True)
        rejected_df.to_csv(reject_csv_path, index=False, encoding="utf-8")

    if review_csv_path:
        Path(review_csv_path).parent.mkdir(parents=True, exist_ok=True)
        review_df.to_csv(review_csv_path, index=False, encoding="utf-8")

    return accepted_df, rejected_df, review_df, classified_df

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

def load_list(path: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#") # ignore comments
        ]
    
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

def pivot_per_contributor_to_per_phd(pairs: pd.DataFrame) -> pd.DataFrame:
    # Group by publication
    aggregated = pairs.groupby(
        [
            "integer_id",
            "thesis_identifier",
            "institution",
            "author_name",
            "title",
            "year",
            "language",
        ],
        dropna=False,
    ).agg(list).reset_index()

    # Make sure the contributor order is a sequence from 1..n_contributors
    aggregated["contributor_order"] = aggregated["contributor_order"].apply(
        lambda lst: list(range(1, len(lst) + 1))
    )

    # Pivot contributors into contributor_1..contributor_n columns
    pubs_list = []
    for _, row in aggregated.iterrows():
        pub_dict = {col: row[col] for col in
                    ["integer_id", "thesis_identifier", "institution", "author_name", "title", "year", "language"]}

        contributors = row["contributor"]
        contributor_orders = row["contributor_order"]

        for order in sorted(set(contributor_orders)):
            idx = order - 1
            if idx < len(contributors):
                pub_dict[f"contributor_{order}"] = contributors[idx]

        pubs_list.append(pub_dict)

    pubs = pd.DataFrame(pubs_list).reset_index(drop=True).convert_dtypes()
    return pubs

