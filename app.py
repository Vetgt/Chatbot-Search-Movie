from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_cpu_model import generate_response
import re

app = Flask(__name__)

print("🔹 Loading IMDb dataset...")

df_titles = pd.read_csv("imdb_data/title.basics.tsv.gz", sep='\t', compression='gzip', low_memory=False)
df_ratings = pd.read_csv("imdb_data/title.ratings.tsv.gz", sep='\t', compression='gzip', low_memory=False)

titles = pd.merge(df_titles, df_ratings, on='tconst')

titles = titles[['primaryTitle', 'startYear', 'genres', 'titleType', 'averageRating', 'numVotes']].dropna()

valid_types = ['movie', 'tvMovie', 'tvSeries', 'short']
titles = titles[titles['titleType'].isin(valid_types)]
titles = titles[titles['numVotes'] >= 500] # Minimal 500 votes
titles['startYear'] = pd.to_numeric(titles['startYear'], errors='coerce')
titles = titles[titles['averageRating'] >= 6.0]

titles['primaryTitle'] = titles['primaryTitle'].str.replace(r'[*\-]', '', regex=True)

unique_genres = set()
for g in titles['genres'].dropna():
    for x in g.split(','):
        unique_genres.add(x.strip())
unique_genres = sorted(list(unique_genres))

print(f"✅ Data Ready! Loaded {len(titles)} clean titles.")

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(titles['genres'].fillna(''))

def recommend_titles(genre_input, year=None):
    input_vec = vectorizer.transform([genre_input])
    sim_scores = cosine_similarity(input_vec, genre_matrix)

    titles['similarity'] = sim_scores[0]
    result = titles.copy()

    if year:
        result = result[(result['startYear'] >= year - 2) & (result['startYear'] <= year + 2)]

    top = result.sort_values(by=['similarity', 'averageRating', 'numVotes'], ascending=False).head(5)
    
    return top[['primaryTitle', 'titleType', 'startYear', 'averageRating']]

def detect_genre_with_llm(user_msg):
    user_msg_lower = user_msg.lower()

    for g in unique_genres:
        if re.search(r'\b' + re.escape(g.lower()) + r'\b', user_msg_lower):
            print(f"✅ Genre detected by Logic: {g}")
            return g

    print("Logic failed")
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
             
    print(f"No genre found in LLM response: {raw_response}")
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')

    year = None
    year_match = re.search(r'\b(19|20)\d{2}\b', user_input)
    if year_match:
        year = int(year_match.group(0))

    detected_genre = detect_genre_with_llm(user_input)

    llm_prompt = (
        f"User said: '{user_input}'.\n"
        "Give a short, friendly conversational reply (max 1 sentence).\n"
        "Do NOT recommend movies yet."
    )
    llm_reply = generate_response(llm_prompt, max_new_tokens=40)

    if detected_genre:
        recs = recommend_titles(detected_genre, year)
        
        if recs.empty:
            year_info = f" around {year}" if year else ""
            reply = f"{llm_reply}\n\nI searched for {detected_genre}{year_info}, but couldn't find good matches."
        else:
            movie_list_text = ""
            for r in recs.itertuples():
                icon = "📺" if "tv" in r.titleType.lower() else "🎬"
                movie_list_text += f"{icon} {r.primaryTitle} ({int(r.startYear)}) — ⭐{r.averageRating}\n"
            
            reply = (
                f"{llm_reply}\n\n"
                f"Here are top {detected_genre}recommendations:\n\n"
                f"{movie_list_text}"
            )
    else:
        reply = (
            f"{llm_reply}\n\n"
            "I'm sorry, I couldn't catch the specific genre. "
            "Try specifying keywords like 'Action, Comedy, or Sci-Fi'."
        )

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)