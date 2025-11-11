import json
import re

# A more general set of common Telugu stopwords
TELUGU_STOPWORDS = {
    'మరియు', 'ఒక', 'లో', 'కి', 'కు', 'నుండి', 'తో', 'యొక్క', 'అనే', 'ఈ', 'ఆ', 'ఏ',
    'అని', 'కోసం', 'ద్వారా', 'కానీ', 'అయితే', 'గురించి', 'మరి', 'కల', 'లేదా', 'కాదు',
    'అవును', 'నేను', 'నువ్వు', 'మేము', 'మీరు', 'అతను', 'ఆమె', 'వారు', 'ఇది', 'అది',
    'ఈయన', 'ఆయన', 'ఉన్నా', 'ఉన్నాయి', 'వద్ద', 'వారి', 'వీరి', 'ఇక్కడ', 'అక్కడ', 'ఎక్కడ',
    'ఎప్పుడు', 'ఎందుకు', 'ఎలా', 'చాలా', 'కొన్ని', 'కంటే', 'కూడా', 'గా', 'చేశారు', 'చేసారు',
    'అప్పుడు', 'ఇప్పుడు', 'ఎందుకంటే', 'మొదలైన', 'తరువాత', 'ఎక్కువ', 'తక్కువ', 'పాటు',
    'లోని', 'పైన', 'క్రింద'
}

# --- NLP Preprocessing Function (Defined but NOT USED in main flow) ---
def nlp_clean_text(text, stopwords):
    """
    Performs standard NLP cleaning: tokenization, stopword removal, and joining.
    """
    if not text:
        return ""
    # 1. TOKENIZE :
    #  Find all sequences of Telugu characters (words).
    tokens = re.findall(r'[\u0c00-\u0c7F]+', text)
    # 2. STOPWORDS REMOVAL: 
    # Create a new list excluding any word in the stopwords set.
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    # 3. JOIN:
    #  Combine the cleaned tokens back into a single string.
    return ' '.join(cleaned_tokens)

# --- Helper Function for Regex Extraction ---
def extract_field(pattern, text):
    """Helper function to extract a field using regex, returns empty string if not found."""
    match = re.search(pattern, text)
    if match:
        # Use strip() to remove any leading/trailing whitespace
        return match.group(1).strip()
    return ""

# --- Main Data Processing Function ---
def preprocess_raw_data(raw_file_path, final_file_path):
    """
    Reads the raw dataset (with 'Label: Value.' format), parses the paragraphs
    using regex to extract structured information, and saves it as the final dataset.
    """
    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            ## utf-8 encoding is to preserve the telugu text properly
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {raw_file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {raw_file_path}.")
        return

    final_dataset = []

    for index, item in enumerate(raw_data): # Added index for debugging if needed
        paragraph = item.get("poet_data_paragraph", "")

        # --- ID Extraction (with safety) ---
        id_str = extract_field(r"ID: (\d+)\.", paragraph)
        poet_id = 0 # Default value
        if id_str:
            try:
                poet_id = int(id_str)
            except ValueError:
                print(f"Warning: Could not convert ID '{id_str}' to int for item {index}. Using 0.")
        # else: poet_id remains 0

        # --- Field Extractions using Regex ---

        # Using the ORIGINAL non-greedy pattern for name as requested
        #--------------NME OF THE POET--------------
        name = extract_field(r"కవి: (.*?)\.", paragraph)
        #---------------TAGS IF ANY THEY HAVE----
        titles = extract_field(r"బిరుదులు: (.*?)\.", paragraph)

        #  patterns for ERA ,BIRTH YEAR,DEATH YEAR-----------
        era = extract_field(r"కాలం: (సా\.శ\..*?)\.", paragraph)
        birth_year = extract_field(r"జననం: (సా\.శ\.\s*\d+)\.", paragraph)
        death_year = extract_field(r"మరణం: (సా\.శ\.\s*\d+)\.", paragraph)
        #----------POETS BIO-----------
        biography = extract_field(r"జీవిత సారాంశం: (.*?)\.\.", paragraph) # Double dot capture
        #---------------PLACE OF BIRTH-------------
        birth_place = extract_field(r"జనన స్థలం: (.*?)\.", paragraph)
        #------------FAMOUS WORKS--------------
        famous_works_str = extract_field(r"ప్రసిద్ధ రచనలు: (.*?)\.", paragraph)

        # Corrected Famous Works Splitting (Handles commas inside parentheses)
        famous_works = [] # Initialize an empty list
        if famous_works_str:
            # Use regex to split by comma, but NOT commas inside parentheses
            split_works = re.split(r',\s*(?![^()]*\))', famous_works_str)
            # Loop through the split parts, strip whitespace, and add non-empty ones
            for work in split_works:
                cleaned_work = work.strip()
                if cleaned_work: # Only add if it's not an empty string after stripping
                    famous_works.append(cleaned_work)
        # else: famous_works remains an empty list

        # Assemble the final structured dictionary
        poet_structured_data = {
            "id": poet_id,
            "name_telugu": name,
            "titles": titles,
            "era": era,
            "biography_summary": biography, # NOTE: NLP cleaning is NOT applied here
            "birth_place_telugu": birth_place,
            "birth_year": birth_year,
            "death_year": death_year,
            "famous_works": famous_works,
            # Poems are copied directly without any processing
            "poems": item.get("poems_raw", [])
        }

        final_dataset.append(poet_structured_data)

    # Save the processed data to the final JSON file
    with open(final_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"Preprocessing complete. Final dataset saved to '{final_file_path}'")

# --- Main execution ---
if __name__ == "__main__":
    # Ensure this matches the EXACT name of your raw data file
    RAW_DATASET_FILE = "raw_dataset.json"
    FINAL_DATASET_FILE = "Final_Dataset_Generated.json"
    preprocess_raw_data(RAW_DATASET_FILE, FINAL_DATASET_FILE)