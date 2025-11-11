import json
import re  # We need regex for the rule-based part
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Define Example Questions for Retrieval (The "Smart" Part) ---
RETRIEVAL_QUESTIONS = {
    "ask_biography": [
        "POET_NAME గురించి చెప్పు",
        "POET_NAME ఎవరు?",
        "POET_NAME జీవితం గురించి సమాచారం ఇవ్వు",
        "POET_NAME బయోగ్రఫీ",
        "Who is POET_NAME?",
        "tell me about POET_NAME"
    ],
    "ask_titles": [
        "POET_NAME బిరుదులు ఏవి?",
        "POET_NAME గారి బిరుదులు",
        "What are POET_NAME's titles?"
    ],
    "ask_famous_works": [
        "POET_NAME రచనలు ఏవి?",
        "POET_NAME రాసిన పుస్తకాలు చెప్పు",
        "POET_NAME ప్రసిద్ధ రచనలు",
        "What did POET_NAME write?"
    ],
    "ask_era": [
        "POET_NAME కాలం ఏది?",
        "POET_NAME ఎప్పుడు జీవించారు?",
        "What is POET_NAME's era?"
    ],
    "ask_birth_place": [
        "POET_NAME ఎక్కడ పుట్టారు?",
        "POET_NAME జనన స్థలం",
        "POET_NAME birthplace"
    ],
    "ask_lifespan": [
        "POET_NAME జననం మరియు మరణం",
        "POET_NAME జీవన కాలం",
        "POET_NAME lifespan"
    ],
    # --- NEW: Added more examples to make this intent smarter ---
    "ask_poem": [
        "POET_NAME పద్యం ఒకటి చెప్పు",
        "POET_NAME నుండి ఒక పద్యం",
        "POET_NAME poem",
        "display poem of POET_NAME",
        "POET_NAME పద్యం చూపించు"
    ]
}


class ChatbotEngine:
    def __init__(self, json_file_path):
        print("Bot Engine: Loading data...")
        self.data_all, self.poets_by_name, self.poets_by_id = self._load_data(json_file_path)
        
        # --- 1. Setup for Retrieval Model ---
        self.documents = []
        self.metadata = []
        
        print("Bot Engine: Building retrieval knowledge base...")
        self._build_retrieval_kb()
        self._save_processed_data_to_files() # Generate files for you
        self._vectorize_kb()
        
        # --- 2. Setup for Rule-Based Model ---
        print("Bot Engine: Compiling rule-based intents...")
        self.rule_based_intents = [
            # --- NEW INTENT 1: Get poem by POET and GENRE ---
            # Example: "'భక్తి నివేదన' genre poem from తిక్కన"
            (re.compile(r"'(.*)' (?:genre|style|శైలి) (?:poem|పద్యం) (?:from|by) (.*)", re.IGNORECASE), self._handle_get_poem_by_genre_and_poet),
            
            # --- NEW INTENT 2: List poets by GENRE ---
            # Example: "'నీతి బోధన' శైలి కవులు ఎవరు?"
            (re.compile(r"'(.*)' (?:genre|style|శైలి)[\w\s]* (?:poets|కవులు|ఎవరు)", re.IGNORECASE), self._handle_list_by_genre),
            
            # --- Existing Rules ---
            (re.compile(r"(కవిత్రయం) (ఎవరు|గురించి)"), self._handle_kavitrayam),
            (re.compile(r"(అష్టదిగ్గజాలు) (ఎవరు|గురించి)"), self._handle_ashtadiggajalu),
            (re.compile(r"(\d{2})\s*(?:va|వ)?\s*(?:శతాబ్ద|satabda)[\w\s]* (?:jabitha|list|కవులు|జాబితా)"), self._handle_list_by_era),
            (re.compile(r"(.*) (సమకాలికులు|సమకాలీన కవులు) ఎవరు"), self._handle_contemporaries),
            (re.compile(r"'(.*)' (రచన|పుస్తకం) ఎవరు రాశారు"), self._handle_find_poet_by_work),
        ]
        
        print(f"Bot Engine: Ready. (Retrieval KB: {len(self.documents)} entries, Rule KB: {len(self.rule_based_intents)} rules)")

    # --- Data Loading and Setup Functions ---

    def _load_data(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_poets = json.load(f)
            poets_by_name = {p['name_telugu'].strip(): p for p in all_poets}
            poets_by_id = {p['id']: p for p in all_poets}
            print(f"డేటా విజయవంతంగా లోడ్ చేయబడింది ({len(all_poets)} కవులు).")
            return all_poets, poets_by_name, poets_by_id
        except Exception as e:
            print(f"లోపం: డేటా లోడ్ చేయడంలో విఫలమైంది: {e}")
            exit()

    def _build_retrieval_kb(self):
        """Builds the KB for the simple, factual intents."""
        current_doc_index = 0
        
        for poet in self.data_all:
            poet_name = poet['name_telugu']
            
            def add_intent(intent_type, answer_text, extra_meta={}):
                nonlocal current_doc_index
                self.documents.append(answer_text)
                answer_meta = {
                    'poet_name': poet_name, 'type': intent_type,
                    'is_answer': True, 'doc_index': current_doc_index
                }
                answer_meta.update(extra_meta)
                self.metadata.append(answer_meta)
                answer_index = current_doc_index
                current_doc_index += 1
                
                # Add all corresponding QUESTIONS from the list
                if intent_type in RETRIEVAL_QUESTIONS:
                    for q in RETRIEVAL_QUESTIONS[intent_type]:
                        self.documents.append(q.replace("POET_NAME", poet_name))
                        self.metadata.append({
                            'is_answer': False, 'points_to_index': answer_index
                        })
                        current_doc_index += 1

            # Add all simple retrieval intents
            add_intent("ask_biography", poet['biography_summary'])
            add_intent("ask_titles", poet['titles'])
            add_intent("ask_famous_works", ", ".join(poet['famous_works']))
            add_intent("ask_era", poet['era'])
            add_intent("ask_birth_place", poet['birth_place_telugu'])
            add_intent("ask_lifespan", f"జననం: {poet['birth_year']}, మరణం: {poet['death_year']}")

            # NEW: We must also add the poems themselves to the retrieval KB
            # so they can be found by the "ask_poem" intent
            for poem in poet.get('poems', []):
                add_intent("ask_poem", poem['text'], extra_meta={'genre': poem['genre']})


    def _save_processed_data_to_files(self):
        """TASK: GENERATE FILES FOR THE ULTIMATE DATASET"""
        print("Bot Engine: Saving retrieval KB to files...")
        with open("knowledge_base_documents.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(self.documents):
                f.write(f"----- DOCUMENT {i} -----\n{doc}\n\n")
        
        with open("knowledge_base_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)
        print("Bot Engine: Files 'knowledge_base_documents.txt' and 'knowledge_base_metadata.json' created.")

    def _vectorize_kb(self):
        """TASK: PROCESS THE DOCUMENTS"""
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.kb_vectors = self.vectorizer.fit_transform(self.documents)

    # --- Rule-Based Handler Functions (from poet_bot.py) ---

    def _find_poet_by_name(self, name_query):
        """Helper to find a poet from a partial name."""
        clean_query = name_query.strip().lower()
        if not clean_query:
            return None
        for full_name, poet_data in self.poets_by_name.items():
            if clean_query in full_name.lower():
                return poet_data
        return None

    def _handle_kavitrayam(self, match):
        names = [self.poets_by_id.get(i, {}).get('name_telugu', f'ID {i} Not Found') for i in [1, 2, 3]]
        return f"కవిత్రయం: {', '.join(names)}."

    def _handle_ashtadiggajalu(self, match):
        search_term = "అష్టదిగ్గజాలు"
        poets = [p['name_telugu'] for p in self.data_all if search_term in p['biography_summary']]
        if poets:
            return f"ఈ డేటాసెట్‌లో అష్టదిగ్గజాలుగా పేర్కొనబడిన కవులు: {', '.join(poets)}"
        return "ఈ డేటాసెట్‌లో 'అష్టదిగ్గజాలు' అని స్పష్టంగా ఎవరూ లేరు."

    def _handle_list_by_era(self, match):
        century_num = match.group(1) # This will be "12"
        search_term = f"{century_num}వ శతాబ్దం" 
        poets = [p['name_telugu'] for p in self.data_all if search_term in p['era']]
        if poets:
            return f"{century_num}వ శతాబ్దపు కవులు: {', '.join(poets)}"
        return f"క్షమించండి, {century_num}వ శతాబ్దానికి చెందిన కవులు ఎవరూ కనబడలేదు."

    def _handle_contemporaries(self, match):
        name = match.group(1).strip()
        target_poet = self._find_poet_by_name(name)
        if not target_poet:
            return f"క్షమించండి, '{name}' అనే కవి కనబడలేదు."
        
        target_era = target_poet['era']
        poets = [p['name_telugu'] for p in self.data_all 
                 if p['era'] == target_era and p['id'] != target_poet['id']]
        
        if poets:
            return f"{target_poet['name_telugu']} గారి సమకాలికులు ({target_era}): {', '.join(poets)}"
        return f"{target_era} కాలానికి చెందిన ఇతర కవులు ఈ డేటాసెట్‌లో లేరు."
        
    def _handle_find_poet_by_work(self, match):
        work_query = match.group(1).strip().lower()
        found_poets = []
        for poet in self.data_all:
            for work in poet['famous_works']:
                if work_query in work.lower():
                    found_poets.append(poet['name_telugu'])
                    break 
        if found_poets:
            return f"'{work_query}' అనే రచనను వీరు రాసారు: {', '.join(found_poets)}"
        return f"క్షమించండి, '{work_query}' రాసిన కవిని కనుగొనలేకపోయాను."

    # --- NEW: Handlers for our new complex intents ---
    
    def _handle_list_by_genre(self, match):
        genre_query = match.group(1).strip().lower()
        found_poets = set() # Use a set to avoid duplicates
        for poet in self.data_all:
            for poem in poet.get('poems', []):
                if genre_query in poem['genre'].lower():
                    found_poets.add(poet['name_telugu'])
                    break # Move to the next poet
        
        if found_poets:
            return f"'{genre_query}' శైలిలో పద్యాలు రాసిన కవులు: {', '.join(found_poets)}"
        return f"క్షమించండి, '{genre_query}' శైలిలో పద్యాలు రాసిన కవులు ఈ డేటాసెట్‌లో కనబడలేదు."

    def _handle_get_poem_by_genre_and_poet(self, match):
        genre_query = match.group(1).strip().lower()
        poet_name = match.group(2).strip()
        
        poet = self._find_poet_by_name(poet_name)
        if not poet:
            return f"క్షమించండి, '{poet_name}' అనే కవి కనబడలేదు."
            
        found_poems = []
        for poem in poet.get('poems', []):
            if genre_query in poem['genre'].lower():
                found_poems.append(f"({poem['genre']}):\n{poem['text']}\n")
        
        if found_poems:
            # Return the first match
            return (f"{poet['name_telugu']} గారి నుండి '{genre_query}' శైలికి చెందిన పద్యం:\n\n" +
                    found_poems[0])
        return f"క్షమించండి, {poet['name_telugu']} గారి నుండి '{genre_query}' శైలికి చెందిన పద్యాలు కనుగొనబడలేదు."


    # --- THE ULTIMATE LOGIC: get_response ---

    def get_response(self, user_question):
        """
        The main Hybrid Logic.
        1. Try to match a specific RULE.
        2. If no rule matches, use the flexible RETRIEVAL model.
        """
        
        # --- 1. Try Rule-Based Matching First ---
        for pattern, handler in self.rule_based_intents:
            match = pattern.search(user_question)
            if match:
                print(f"--- (Debug: Matched Rule: {pattern.pattern}) ---")
                return handler(match)
        
        # --- 2. If No Rule Matched, Use Retrieval Model ---
        question_vector = self.vectorizer.transform([user_question])
        scores = cosine_similarity(question_vector, self.kb_vectors)
        best_index = scores.argmax()
        best_score = scores[0, best_index]
        
        if best_score < 0.25:
            return "క్షమించండి, మీ ప్రశ్న నాకు అర్థం కాలేదు. దయచేసి మరో విధంగా అడగగలరు."
        
        match_meta = self.metadata[best_index]
        
        if match_meta['is_answer']:
            answer_index = match_meta['doc_index']
        else:
            answer_index = match_meta['points_to_index']
            
        answer_text = self.documents[answer_index]
        answer_meta = self.metadata[answer_index]
        
        # --- 3. Format the Retrieved Answer ---
        poet = answer_meta['poet_name']
        type = answer_meta['type']
        
        if type == 'ask_biography':
            return f"{poet} గారి గురించి ఇక్కడ కొంత సమాచారం ఉంది: \n{answer_text}"
        elif type == 'ask_titles':
            return f"{poet} గారి బిరుదులు: \n{answer_text}"
        elif type == 'ask_famous_works':
            return f"{poet} గారి ప్రసిద్ధ రచనలు: \n{answer_text}"
        elif type == 'ask_era':
            return f"{poet} గారి కాలం: \n{answer_text}"
        elif type == 'ask_birth_place':
            return f"{poet} గారి జనన స్థలం: \n{answer_text}"
        elif type == 'ask_lifespan':
            return f"{poet} గారి జీవన కాలం: \n{answer_text}"
        # This now handles the simple "display poem of Nannaya" request
        elif type == 'ask_poem':
            return f"{poet} గారి పద్యం (శైలి: {answer_meta['genre']}): \n{answer_text}"
        
        return answer_text