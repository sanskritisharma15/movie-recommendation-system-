from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Your OMDb API key
OMDB_API_KEY = "ddd0246e"

POSTER_CACHE = {}

# Use a dedicated session that ignores proxy environment variables.
# (Fixes hangs/timeouts caused by stray HTTP_PROXY/HTTPS_PROXY settings.)
SESSION = requests.Session()
SESSION.trust_env = False

if not OMDB_API_KEY:
    print("=" * 60)
    print("⚠️  WARNING: OMDb API key not set!")
    print("Posters will NOT load until you add a key.")
    print("Get a free key at: https://www.omdbapi.com/apikey.aspx")
    print("Then paste it into OMDB_API_KEY in app.py")
    print("=" * 60)


def get_poster(title):
    """Fetch real poster from OMDb API, cache result, fallback to None on failure."""
    if title in POSTER_CACHE:
        return POSTER_CACHE[title]

    poster_url = None
    if OMDB_API_KEY:
        try:
            resp = SESSION.get(
                "http://www.omdbapi.com/",
                params={"t": title, "apikey": OMDB_API_KEY},
                timeout=6
            )
            data = resp.json()
            if data.get("Response") == "True" and data.get("Poster") not in (None, "N/A", ""):
                poster_url = data["Poster"]
            else:
                print(f"[poster] No poster found for '{title}': {data.get('Error', 'no Poster field')}")
        except requests.RequestException as e:
            print(f"[poster] Request failed for '{title}': {e}")

    POSTER_CACHE[title] = poster_url
    return poster_url


# ---------- Genre-wise movie lists: (title, industry) ----------

action = [
    ("The Dark Knight", "Hollywood"), ("Iron Man", "Hollywood"), ("The Avengers", "Hollywood"),
    ("Thor", "Hollywood"), ("John Wick", "Hollywood"), ("Mission: Impossible - Fallout", "Hollywood"),
    ("Die Hard", "Hollywood"), ("Mad Max: Fury Road", "Hollywood"), ("Gladiator", "Hollywood"),
    ("The Matrix", "Hollywood"), ("Terminator 2: Judgment Day", "Hollywood"), ("Skyfall", "Hollywood"),
    ("The Bourne Identity", "Hollywood"), ("Black Panther", "Hollywood"),
    ("Captain America: Civil War", "Hollywood"), ("Top Gun: Maverick", "Hollywood"),
    ("Kill Bill: Vol. 1", "Hollywood"), ("300", "Hollywood"), ("Wonder Woman", "Hollywood"),
    ("Spider-Man: No Way Home", "Hollywood"),
    ("War", "Bollywood"), ("Pathaan", "Bollywood"), ("Tiger Zinda Hai", "Bollywood"),
    ("Dhoom", "Bollywood"), ("Dhoom 2", "Bollywood"), ("Singham", "Bollywood"),
    ("Krrish", "Bollywood"), ("Bang Bang", "Bollywood"), ("Baby", "Bollywood"),
    ("Uri: The Surgical Strike", "Bollywood"), ("Don", "Bollywood"), ("Ghajini", "Bollywood"),
    ("Baahubali: The Beginning", "Tollywood"), ("RRR", "Tollywood"), ("Pushpa: The Rise", "Tollywood"),
    ("Saaho", "Tollywood"), ("Sye Raa Narasimha Reddy", "Tollywood"), ("Magadheera", "Tollywood"),
    ("Temper", "Tollywood"),
]

romance = [
    ("The Notebook", "Hollywood"), ("La La Land", "Hollywood"), ("Titanic", "Hollywood"),
    ("A Walk to Remember", "Hollywood"), ("The Fault in Our Stars", "Hollywood"),
    ("Pride & Prejudice", "Hollywood"), ("Me Before You", "Hollywood"), ("About Time", "Hollywood"),
    ("500 Days of Summer", "Hollywood"), ("Before Sunrise", "Hollywood"), ("The Vow", "Hollywood"),
    ("Crazy Rich Asians", "Hollywood"), ("To All the Boys I've Loved Before", "Hollywood"),
    ("Kal Ho Naa Ho", "Bollywood"), ("Jab We Met", "Bollywood"), ("Rockstar", "Bollywood"),
    ("Ae Dil Hai Mushkil", "Bollywood"), ("Kabhi Khushi Kabhie Gham", "Bollywood"),
    ("Dilwale Dulhania Le Jayenge", "Bollywood"), ("Yeh Jawaani Hai Deewani", "Bollywood"),
    ("Tamasha", "Bollywood"), ("Barfi!", "Bollywood"), ("Veer-Zaara", "Bollywood"),
    ("Raanjhanaa", "Bollywood"),
    ("Arjun Reddy", "Tollywood"), ("Geetha Govindam", "Tollywood"), ("Ninnu Kori", "Tollywood"),
    ("Fidaa", "Tollywood"), ("Yeh Maaya Chesave", "Tollywood"), ("Sita Ramam", "Tollywood"),
]

horror = [
    ("The Conjuring", "Hollywood"), ("Insidious", "Hollywood"), ("Annabelle", "Hollywood"),
    ("Hereditary", "Hollywood"), ("The Exorcist", "Hollywood"), ("A Nightmare on Elm Street", "Hollywood"),
    ("It", "Hollywood"), ("The Ring", "Hollywood"), ("Sinister", "Hollywood"), ("Get Out", "Hollywood"),
    ("A Quiet Place", "Hollywood"), ("The Nun", "Hollywood"), ("Paranormal Activity", "Hollywood"),
    ("Halloween", "Hollywood"), ("The Babadook", "Hollywood"), ("The Shining", "Hollywood"),
    ("Evil Dead", "Hollywood"), ("Scream", "Hollywood"), ("Poltergeist", "Hollywood"),
    ("Stree", "Bollywood"), ("Bhoot", "Bollywood"), ("Tumbbad", "Bollywood"), ("Raaz", "Bollywood"),
    ("1920", "Bollywood"), ("Pari", "Bollywood"), ("Bhool Bhulaiyaa", "Bollywood"),
    ("Stree 2", "Bollywood"),
    ("Arundhati", "Tollywood"), ("Raju Gari Gadhi", "Tollywood"), ("Awe!", "Tollywood"),
]

comedy = [
    ("Superbad", "Hollywood"), ("The Hangover", "Hollywood"), ("Dumb and Dumber", "Hollywood"),
    ("Anchorman", "Hollywood"), ("Bridesmaids", "Hollywood"), ("Step Brothers", "Hollywood"),
    ("21 Jump Street", "Hollywood"), ("Zombieland", "Hollywood"), ("Mean Girls", "Hollywood"),
    ("Ferris Bueller's Day Off", "Hollywood"), ("Deadpool", "Hollywood"),
    ("Jumanji: Welcome to the Jungle", "Hollywood"), ("Game Night", "Hollywood"),
    ("3 Idiots", "Bollywood"), ("Zindagi Na Milegi Dobara", "Bollywood"), ("Dil Chahta Hai", "Bollywood"),
    ("Hera Pheri", "Bollywood"), ("Golmaal", "Bollywood"), ("Andaz Apna Apna", "Bollywood"),
    ("Chupke Chupke", "Bollywood"), ("Welcome", "Bollywood"), ("Munna Bhai M.B.B.S.", "Bollywood"),
    ("Bhool Bhulaiyaa 2", "Bollywood"), ("Housefull", "Bollywood"),
    ("Ala Vaikunthapurramuloo", "Tollywood"), ("F2: Fun and Frustration", "Tollywood"),
    ("Jathi Ratnalu", "Tollywood"), ("Bhale Bhale Magadivoy", "Tollywood"),
    ("Middle Class Abbayi", "Tollywood"), ("Son of Satyamurthy", "Tollywood"),
]

motivational = [
    ("The Pursuit of Happyness", "Hollywood"), ("Rocky", "Hollywood"), ("Rudy", "Hollywood"),
    ("The Blind Side", "Hollywood"), ("Invictus", "Hollywood"), ("Coach Carter", "Hollywood"),
    ("Miracle", "Hollywood"), ("Remember the Titans", "Hollywood"), ("The Karate Kid", "Hollywood"),
    ("Cinderella Man", "Hollywood"), ("Soul Surfer", "Hollywood"), ("Hidden Figures", "Hollywood"),
    ("Moneyball", "Hollywood"), ("42", "Hollywood"),
    ("Bhaag Milkha Bhaag", "Bollywood"), ("Dangal", "Bollywood"), ("Mary Kom", "Bollywood"),
    ("Chak De India", "Bollywood"), ("83", "Bollywood"), ("Taare Zameen Par", "Bollywood"),
    ("Iqbal", "Bollywood"), ("Lagaan", "Bollywood"), ("Gold", "Bollywood"), ("Toofaan", "Bollywood"),
    ("Jersey", "Tollywood"), ("Sye", "Tollywood"),
]

biopic = [
    ("The Social Network", "Hollywood"), ("Steve Jobs", "Hollywood"), ("Bohemian Rhapsody", "Hollywood"),
    ("The Theory of Everything", "Hollywood"), ("A Beautiful Mind", "Hollywood"), ("Ray", "Hollywood"),
    ("Walk the Line", "Hollywood"), ("The Imitation Game", "Hollywood"), ("Lincoln", "Hollywood"),
    ("Ford v Ferrari", "Hollywood"), ("Catch Me If You Can", "Hollywood"), ("J. Edgar", "Hollywood"),
    ("Gandhi", "Hollywood"),
    ("Sanju", "Bollywood"), ("Neerja", "Bollywood"), ("Shakuntala Devi", "Bollywood"),
    ("Manikarnika: The Queen of Jhansi", "Bollywood"), ("Super 30", "Bollywood"), ("Pad Man", "Bollywood"),
    ("Mahanati", "Tollywood"), ("NTR: Kathanayakudu", "Tollywood"), ("Yatra", "Tollywood"),
]

sad_drama = [
    ("Forrest Gump", "Hollywood"), ("The Shawshank Redemption", "Hollywood"),
    ("Schindler's List", "Hollywood"), ("Green Book", "Hollywood"), ("Manchester by the Sea", "Hollywood"),
    ("A Star is Born", "Hollywood"), ("The Pianist", "Hollywood"), ("Life is Beautiful", "Hollywood"),
    ("Marriage Story", "Hollywood"), ("Room", "Hollywood"),
    ("Kabhi Alvida Naa Kehna", "Bollywood"), ("Kabir Singh", "Bollywood"), ("Devdas", "Bollywood"),
    ("Highway", "Bollywood"), ("Masaan", "Bollywood"), ("Piku", "Bollywood"), ("October", "Bollywood"),
    ("Kapoor & Sons", "Bollywood"),
    ("Uppena", "Tollywood"), ("Rangasthalam", "Tollywood"), ("C/o Kancharapalem", "Tollywood"),
    ("Malli Raava", "Tollywood"),
]

genre_meta = {
    "Action":       {"keywords": "action fight battle war explosive intense chase", "key": "action"},
    "Romance":      {"keywords": "romance love emotional relationship heartfelt passion", "key": "romance"},
    "Horror":       {"keywords": "horror scary supernatural fear dark haunted", "key": "horror"},
    "Comedy":       {"keywords": "comedy funny humor light-hearted laugh silly", "key": "comedy"},
    "Motivational": {"keywords": "motivational inspiring struggle success uplifting determination", "key": "motivational"},
    "Biopic":       {"keywords": "biopic real-life inspiring true-story journey achievement", "key": "biopic"},
    "Sad/Drama":    {"keywords": "drama emotional sad tragic heartbreaking loss", "key": "drama"},
}

genre_lists = {
    "Action": action, "Romance": romance, "Horror": horror, "Comedy": comedy,
    "Motivational": motivational, "Biopic": biopic, "Sad/Drama": sad_drama,
}

rows = []
for genre, movies in genre_lists.items():
    meta = genre_meta[genre]
    for title, industry in movies:
        rows.append({
            "title": title,
            "industry": industry,
            "genre": genre,
            "genre_key": meta["key"],
            "description": meta["keywords"]
        })

df = pd.DataFrame(rows).drop_duplicates(subset="title").reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["description"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

GENRE_PAGE_SIZE = 6  # how many posters shown per genre box initially


def recommend(movie_name, top_n=5):
    movie_name = movie_name.strip().lower()
    titles_lower = df["title"].str.lower()

    if movie_name not in titles_lower.values:
        return None

    idx = titles_lower[titles_lower == movie_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    results = []
    for i, score in scores:
        title = df["title"][i]
        results.append({
            "title": title,
            "genre": df["genre"][i],
            "genre_key": df["genre_key"][i],
            "industry": df["industry"][i],
            "score": round(float(score) * 100),
            "poster": get_poster(title)
        })
    return results


def build_genre_sections():
    sections = []
    for genre_name in genre_lists.keys():
        genre_key = genre_meta[genre_name]["key"]
        genre_df = df[df["genre"] == genre_name].reset_index(drop=True)
        initial = genre_df.iloc[:GENRE_PAGE_SIZE][["title", "genre", "genre_key", "industry"]].to_dict("records")
        for m in initial:
            m["poster"] = get_poster(m["title"])
        sections.append({
            "name": genre_name,
            "key": genre_key,
            "total": len(genre_df),
            "movies": initial
        })
    return sections


@app.route("/", methods=["GET", "POST"])
def index():
    movies = df["title"].tolist()
    recommendations = None
    selected_movie = None
    selected_movie_data = None
    error = None

    if request.method == "POST":
        selected_movie = request.form.get("movie")
        recommendations = recommend(selected_movie)
        if recommendations is None:
            error = f"'{selected_movie}' not found. Please pick from the list."
        else:
            row = df[df["title"] == selected_movie].iloc[0]
            selected_movie_data = {
                "title": row["title"],
                "genre": row["genre"],
                "genre_key": row["genre_key"],
                "industry": row["industry"],
                "poster": get_poster(row["title"])
            }

    genre_sections = build_genre_sections()

    return render_template(
        "index.html",
        movies=movies,
        genre_sections=genre_sections,
        page_size=GENRE_PAGE_SIZE,
        recommendations=recommendations,
        selected_movie=selected_movie,
        selected_movie_data=selected_movie_data,
        error=error
    )


@app.route("/load-more-genre")
def load_more_genre():
    genre_name = request.args.get("genre", "")
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", GENRE_PAGE_SIZE))

    genre_df = df[df["genre"] == genre_name].reset_index(drop=True)
    batch = genre_df.iloc[offset: offset + limit][["title", "genre", "genre_key", "industry"]].to_dict("records")
    for m in batch:
        m["poster"] = get_poster(m["title"])
    return jsonify(batch)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)