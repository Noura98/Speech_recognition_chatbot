import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load and preprocess corpus
def preprocess(text_file='data/alice.txt'):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    return paragraphs

# Corpus
original_paragraphs = preprocess()

# Embedder
model = SentenceTransformer('all-MiniLM-L6-v2')
paragraph_embeddings = model.encode(original_paragraphs, convert_to_tensor=True)

# Keyword filtering + semantic scoring
def get_most_relevant_paragraph_hybrid(user_query):
    query_words = set(word_tokenize(user_query.lower())) - stop_words

    # Step 1: Filter candidates using keyword match
    keyword_candidates = []
    for i, para in enumerate(original_paragraphs):
        if any(qw in para.lower() for qw in query_words):
            keyword_candidates.append((i, para))

    # Fallback: use all paragraphs if no keyword matches
    if not keyword_candidates:
        keyword_candidates = list(enumerate(original_paragraphs))

    # Step 2: Apply semantic similarity
    candidate_indices = [idx for idx, _ in keyword_candidates]
    candidate_texts = [para for _, para in keyword_candidates]
    candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    print(f"[DEBUG] Best hybrid score: {best_score:.4f}")

    if best_score < 0.3:
        return "Sorry, I couldn't find a relevant answer."

    return candidate_texts[best_idx]

# Chatbot function
def chatbot(user_input):
    return get_most_relevant_paragraph_hybrid(user_input)
