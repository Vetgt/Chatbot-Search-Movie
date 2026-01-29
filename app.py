from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_cpu_model import generate_response
import re

app = Flask(__name__)

# --- 1. LOAD DATA ---
print("🔹 Loading IMDb dataset... (Please wait)")

try:
    df_titles = pd.read_csv("imdb_data/title.basics.tsv.gz", sep='\t', compression='gzip', low_memory=False)
    df_ratings = pd.read_csv("imdb_data/title.ratings.tsv.gz", sep='\t', compression='gzip', low_memory=False)
except FileNotFoundError:
    print(" ERROR: Dataset not found! Make sure files are in 'imdb_data/' folder.")
    exit()

# Merge Data
titles = pd.merge(df_titles, df_ratings, on='tconst')
titles = titles[['tconst', 'primaryTitle', 'startYear', 'genres', 'titleType', 'averageRating', 'numVotes']].dropna()

# Filter Data (Movies/TV only, Votes > 500, Rating > 6.0)
valid_types = ['movie', 'tvMovie', 'tvSeries', 'short']
titles = titles[titles['titleType'].isin(valid_types)]
titles = titles[titles['numVotes'] >= 500] 
titles['startYear'] = pd.to_numeric(titles['startYear'], errors='coerce')
titles = titles[titles['averageRating'] >= 6.0]

# Clean Titles
titles['primaryTitle'] = titles['primaryTitle'].str.replace(r'[*\-]', '', regex=True)

# Create Unique Genre List
unique_genres = set()
for g in titles['genres'].dropna():
    for x in g.split(','):
        unique_genres.add(x.strip())
unique_genres = sorted(list(unique_genres))

print(f" Data Ready! {len(titles)} titles loaded.")

# --- 2. NLP SETUP ---
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(titles['genres'].fillna(''))

# --- 3. CORE FUNCTIONS ---

def recommend_titles(genre_input, year=None):
    input_vec = vectorizer.transform([genre_input])
    sim_scores = cosine_similarity(input_vec, genre_matrix)

    titles['similarity'] = sim_scores[0]
    result = titles.copy()

    if year:
        result = result[(result['startYear'] >= year - 2) & (result['startYear'] <= year + 2)]

    top = result.sort_values(by=['similarity', 'averageRating', 'numVotes'], ascending=False).head(5)
    
    # Return 'tconst' for IMDb links
    return top[['tconst', 'primaryTitle', 'titleType', 'startYear', 'averageRating']]

def detect_genre_with_llm(user_msg):
    """Detects genre using Mood Dictionary, Regex, or LLM"""
    user_msg_lower = user_msg.lower()

    # --- A. MOOD MAPPING (English & Indonesian Support) ---
    mood_map = {
        # Sadness -> Drama / Romance
        "sad": "Drama", "cry": "Drama", "depressing": "Drama", "tearjerker": "Drama",
        "heartbroken": "Romance", "love": "Romance", "romantic": "Romance",
        "sedih": "Drama", "galau": "Romance", # Indonesian Fallback
        
        # Funny -> Comedy
        "funny": "Comedy", "hilarious": "Comedy", "laugh": "Comedy", "silly": "Comedy",
        "lucu": "Comedy", "ngakak": "Comedy", # Indonesian Fallback
        
        # Scary -> Horror / Thriller
        "scary": "Horror", "spooky": "Horror", "ghost": "Horror", "fear": "Horror",
        "tense": "Thriller", "suspense": "Thriller", "thriller": "Thriller",
        "seram": "Horror", "hantu": "Horror", # Indonesian Fallback
        
        # Action
        "fight": "Action", "battle": "Action", "war": "War", "explosion": "Action",
        "aksi": "Action", "berantem": "Action", # Indonesian Fallback
        
        # Sci-Fi / Fantasy
        "alien": "Sci-Fi", "space": "Sci-Fi", "robot": "Sci-Fi", "future": "Sci-Fi",
        "magic": "Fantasy", "wizard": "Fantasy",
        "cartoon": "Animation", "anime": "Animation"
    }

    # Check Mood Map
    for keyword, genre_target in mood_map.items():
        if keyword in user_msg_lower:
            # Validate if genre exists in DB
            for g in unique_genres:
                if g.lower() == genre_target.lower():
                    print(f" Genre detected by Mood Map: {keyword} -> {g}")
                    return g

    # --- B. Direct Match ---
    for g in unique_genres:
        if re.search(r'\b' + re.escape(g.lower()) + r'\b', user_msg_lower):
            print(f" Genre detected by Direct Match: {g}")
            return g

    # --- C. LLM Fallback ---
    print(" Logic failed, asking LLM...")
    prompt = (
        f"Task: Extract the movie genre from: '{user_msg}'.\n"
        f"List: {', '.join(unique_genres)}\n"
        "Output ONLY the genre name from the list. If unsure, say None."
    )
    
    raw_response = generate_response(prompt, max_new_tokens=15)
    for g in unique_genres:
        if g.lower() in raw_response.lower():
             print(f" Genre detected by LLM: {g}")
             return g
             
    return None

# --- 4. FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')

    # 1. Year Detection
    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', user_input)
    if year_match:
        year = int(year_match.group(0))

    # 2. Genre Detection
    detected_genre = detect_genre_with_llm(user_input)

    # 3. LLM Chit-Chat Generation
    llm_prompt = (
        f"User said: '{user_input}'.\n"
        "Give a short, friendly conversational reply in English (max 1 sentence).\n"
        "Do NOT recommend movies yet."
    )
    llm_reply = generate_response(llm_prompt, max_new_tokens=40)

    # 4. Recommendation Logic
    if detected_genre:
        recs = recommend_titles(detected_genre, year)
        
        if recs.empty:
            year_info = f" around {year}" if year else ""
            reply = f"{llm_reply}<br><br>I searched for <b>{detected_genre}</b>{year_info}, but couldn't find good matches in my database."
        else:
            movie_list_text = ""
            for r in recs.itertuples():
                icon = "📺" if "tv" in str(r.titleType).lower() else "🎬"
                
                # Create IMDb Link
                imdb_link = f"https://www.imdb.com/title/{r.tconst}/"
                
                movie_list_text += (
                    f"<div style='margin-bottom: 8px;'>"
                    f"{icon} <a href='{imdb_link}' target='_blank' style='color: #f5c518; text-decoration: none; font-weight: bold;'>{r.primaryTitle}</a> "
                    f"({int(r.startYear)}) — ⭐{r.averageRating}"
                    f"</div>"
                )
            
            reply = (
                f"{llm_reply}<br><br>"
                f"Here are the top recommendations for <b>{detected_genre}</b>:<br><br>"
                f"{movie_list_text}"
            )
    else:
        reply = (
            f"{llm_reply}<br><br>"
            "I'm sorry, I couldn't catch the specific genre. "
            "Try specifying keywords like <i>'Action', 'Comedy', 'Horror'</i> or describe your mood (e.g., 'sad', 'funny')."
        )

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
