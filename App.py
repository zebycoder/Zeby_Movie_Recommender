import pickle
import numpy as np
import streamlit as st
import requests
import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import json

# Install required package if needed
try:
    from streamlit_lottie import st_lottie
except ImportError:
    import subprocess

    subprocess.run(["pip", "install", "streamlit-lottie"])
    from streamlit_lottie import st_lottie

# Set page config FIRST
st.set_page_config(
    page_title="Zeby Coder's Movie Magic",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- ANIMATION DATA ---- (embedded directly)
film_animation = {
    "v": "5.10.0",
    "fr": 30,
    "ip": 0,
    "op": 90,
    "w": 800,
    "h": 600,
    "nm": "Film Roll",
    "ddd": 0,
    "assets": [],
    "layers": [
        {
            "ddd": 0,
            "ind": 1,
            "ty": 4,
            "nm": "Film Layer",
            "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100, "ix": 11},
                "r": {"a": 1,
                      "k": [{"i": {"x": [0.833], "y": [0.833]}, "o": {"x": [0.167], "y": [0.167]}, "t": 0, "s": [0]},
                            {"t": 90, "s": [360]}], "ix": 10},
                "p": {"a": 0, "k": [400, 300, 0], "ix": 2},
                "a": {"a": 0, "k": [0, 0, 0], "ix": 1},
                "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
            },
            "ao": 0,
            "shapes": [
                {
                    "ty": "gr",
                    "it": [
                        {
                            "ty": "rc",
                            "d": 1,
                            "s": {"a": 0, "k": [500, 100], "ix": 2},
                            "p": {"a": 0, "k": [0, 0], "ix": 3},
                            "r": {"a": 0, "k": 0, "ix": 4},
                            "nm": "Rectangle",
                            "mn": "ADBE Vector Shape - Rect",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.2, 0.2, 0.2, 1], "ix": 4},
                            "o": {"a": 0, "k": 100, "ix": 5},
                            "r": 1,
                            "nm": "Fill",
                            "mn": "ADBE Vector Graphic - Fill",
                            "hd": False
                        },
                        {
                            "ty": "tr",
                            "p": {"a": 0, "k": [0, 0], "ix": 2},
                            "a": {"a": 0, "k": [0, 0], "ix": 1},
                            "s": {"a": 0, "k": [100, 100], "ix": 3},
                            "r": {"a": 0, "k": 0, "ix": 6},
                            "o": {"a": 0, "k": 100, "ix": 7},
                            "sk": {"a": 0, "k": 0, "ix": 4},
                            "sa": {"a": 0, "k": 0, "ix": 5},
                            "nm": "Transform"
                        }
                    ],
                    "nm": "Film Base",
                    "np": 3,
                    "cix": 2,
                    "ix": 1,
                    "mn": "ADBE Vector Group",
                    "hd": False
                },
                {
                    "ty": "el",
                    "d": 1,
                    "p": {"a": 0, "k": [-180, 0], "ix": 3},
                    "s": {"a": 0, "k": [30, 30], "ix": 2},
                    "nm": "Ellipse",
                    "mn": "ADBE Vector Shape - Ellipse",
                    "hd": False
                },
                {
                    "ty": "el",
                    "d": 1,
                    "p": {"a": 0, "k": [-90, 0], "ix": 3},
                    "s": {"a": 0, "k": [30, 30], "ix": 2},
                    "nm": "Ellipse",
                    "mn": "ADBE Vector Shape - Ellipse",
                    "hd": False
                },
                {
                    "ty": "el",
                    "d": 1,
                    "p": {"a": 0, "k": [0, 0], "ix": 3},
                    "s": {"a": 0, "k": [30, 30], "ix": 2},
                    "nm": "Ellipse",
                    "mn": "ADBE Vector Shape - Ellipse",
                    "hd": False
                },
                {
                    "ty": "el",
                    "d": 1,
                    "p": {"a": 0, "k": [90, 0], "ix": 3},
                    "s": {"a": 0, "k": [30, 30], "ix": 2},
                    "nm": "Ellipse",
                    "mn": "ADBE Vector Shape - Ellipse",
                    "hd": False
                },
                {
                    "ty": "el",
                    "d": 1,
                    "p": {"a": 0, "k": [180, 0], "ix": 3},
                    "s": {"a": 0, "k": [30, 30], "ix": 2},
                    "nm": "Ellipse",
                    "mn": "ADBE Vector Shape - Ellipse",
                    "hd": False
                }
            ],
            "ip": 0,
            "op": 90,
            "st": 0,
            "bm": 0
        }
    ],
    "markers": []
}

contact_animation = {
    "v": "5.10.0",
    "fr": 30,
    "ip": 0,
    "op": 90,
    "w": 800,
    "h": 600,
    "nm": "Contact Card",
    "ddd": 0,
    "assets": [],
    "layers": [
        {
            "ddd": 0,
            "ind": 1,
            "ty": 4,
            "nm": "Card Layer",
            "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100, "ix": 11},
                "r": {"a": 0, "k": 0, "ix": 10},
                "p": {"a": 1,
                      "k": [{"i": {"x": 0.833, "y": 0.833}, "o": {"x": 0.167, "y": 0.167}, "t": 0, "s": [400, 300, 0]},
                            {"t": 30, "s": [400, 280, 0]}], "ix": 2},
                "a": {"a": 0, "k": [0, 0, 0], "ix": 1},
                "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
            },
            "ao": 0,
            "shapes": [
                {
                    "ty": "gr",
                    "it": [
                        {
                            "ty": "rc",
                            "d": 1,
                            "s": {"a": 0, "k": [300, 200], "ix": 2},
                            "p": {"a": 0, "k": [0, 0], "ix": 3},
                            "r": {"a": 0, "k": 20, "ix": 4},
                            "nm": "Rectangle",
                            "mn": "ADBE Vector Shape - Rect",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [1, 1, 1, 1], "ix": 4},
                            "o": {"a": 0, "k": 100, "ix": 5},
                            "r": 1,
                            "nm": "Fill",
                            "mn": "ADBE Vector Graphic - Fill",
                            "hd": False
                        },
                        {
                            "ty": "tr",
                            "p": {"a": 0, "k": [0, 0], "ix": 2},
                            "a": {"a": 0, "k": [0, 0], "ix": 1},
                            "s": {"a": 0, "k": [100, 100], "ix": 3},
                            "r": {"a": 0, "k": 0, "ix": 6},
                            "o": {"a": 0, "k": 100, "ix": 7},
                            "sk": {"a": 0, "k": 0, "ix": 4},
                            "sa": {"a": 0, "k": 0, "ix": 5},
                            "nm": "Transform"
                        }
                    ],
                    "nm": "Card",
                    "np": 3,
                    "cix": 2,
                    "ix": 1,
                    "mn": "ADBE Vector Group",
                    "hd": False
                }
            ],
            "ip": 0,
            "op": 90,
            "st": 0,
            "bm": 0
        }
    ],
    "markers": []
}

# ---- STYLES ----
st.markdown(f"""
<style>
    /* Animated gradient background */
    .stApp {{
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
    }}
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Header styles with animation */
    .brand-header {{
        color: white;
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        animation: fadeIn 1s ease-in-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .brand-subheader {{
        color: #FFD700;
        font-family: 'Pacifico', cursive;
        text-align: center;
        margin-top: 0;
        font-size: 1.8rem;
        animation: slideIn 1s ease-out 0.3s both;
    }}
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Contact card with animation */
    .contact-card {{
        background: rgba(255, 255, 255, 0.85);
        border-radius: 25px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform: perspective(500px) rotateX(0deg);
        transition: all 0.5s ease;
        animation: cardEntry 1s ease-out 0.5s both;
    }}
    .contact-card:hover {{
        transform: perspective(500px) rotateX(5deg) translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }}
    @keyframes cardEntry {{
        from {{ opacity: 0; transform: scale(0.9); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}

    /* Recommendation button with pulse animation */
    .recommend-btn {{
        background: linear-gradient(45deg, #FF4B2B, #FF416C) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(255, 65, 108, 0.4) !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        animation: pulse 2s infinite;
    }}
    .recommend-btn:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(255, 65, 108, 0.6) !important;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}

    /* Movie cards with animation */
    .movie-card {{
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        background: rgba(255,255,255,0.9);
        animation: cardAppear 0.6s ease-out;
    }}
    .movie-card:hover {{
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }}
    @keyframes cardAppear {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Footer with animation */
    .footer {{
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 20px;
        border-radius: 15px 15px 0 0;
        text-align: center;
        margin-top: 40px;
        backdrop-filter: blur(8px);
        animation: fadeInUp 1s ease-out;
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(#FF416C, #FF4B2B);
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(#FF4B2B, #FF416C);
    }}
</style>
""", unsafe_allow_html=True)

# Configuration
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
POSTER_PLACEHOLDER = "https://via.placeholder.com/500x750?text=Poster+Not+Available"
DATA_FILES = {
    'movies': 'tmdb_5000_movies.csv',
    'credits': 'tmdb_5000_credits.csv',
    'movie_list': 'movie_list.pkl',
    'similarity': 'similarity.pkl'
}


# Helper functions
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []


def get_director(crew):
    for person in ast.literal_eval(crew):
        if person['job'] == 'Director':
            return person['name']
    return np.nan


def collapse(L):
    return [i.replace(" ", "") for i in L]


@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else POSTER_PLACEHOLDER
    except Exception as e:
        st.error(f"Error fetching poster: {str(e)}")
        return POSTER_PLACEHOLDER


@st.cache_data
def process_data():
    movies_df = pd.read_csv(DATA_FILES['movies'])
    credits_df = pd.read_csv(DATA_FILES['credits'])

    movies_df = movies_df.merge(credits_df, on='title')
    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies_df.dropna(inplace=True)

    movies_df['genres'] = movies_df['genres'].apply(convert)
    movies_df['keywords'] = movies_df['keywords'].apply(convert)
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
    movies_df['director'] = movies_df['crew'].apply(get_director)
    movies_df.dropna(subset=['director'], inplace=True)

    movies_df['genres'] = movies_df['genres'].apply(collapse)
    movies_df['keywords'] = movies_df['keywords'].apply(collapse)
    movies_df['cast'] = movies_df['cast'].apply(collapse)
    movies_df['director'] = movies_df['director'].apply(lambda x: [x.replace(" ", "")])

    movies_df['tags'] = movies_df.apply(lambda row: ' '.join(row['overview'].split()) + ' ' +
                                                    ' '.join(row['genres']) + ' ' +
                                                    ' '.join(row['keywords']) + ' ' +
                                                    ' '.join(row['cast']) + ' ' +
                                                    ' '.join(row['director']), axis=1)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies_df, similarity


def recommend(movie, _movies, _similarity, similarity_threshold=0.1):  # Default set to 0.1
    try:
        index = _movies[_movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(_similarity[index])), reverse=True, key=lambda x: x[1])

        recommended_movie_names = []
        recommended_movie_posters = []

        for i in distances[1:6]:
            if i[1] < similarity_threshold:
                continue
            movie_id = _movies.iloc[i[0]].movie_id
            poster = fetch_poster(movie_id)
            recommended_movie_posters.append(poster)
            recommended_movie_names.append(_movies.iloc[i[0]].title)

        return recommended_movie_names, recommended_movie_posters

    except IndexError:
        st.error("Movie not found in database. Please try another title.")
        return [], []
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return [], []


def main():
    # Header with Animation
    st.markdown('<h1 class="brand-header">🎬 Zeby Coder Presents</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="brand-subheader">AI-Powered Movie Magic</h2>', unsafe_allow_html=True)

    # Load animation directly (embedded in the code)
    st_lottie(film_animation, height=200, key="film-anim")

    # Contact Card in Sidebar
    with st.sidebar:
        st_lottie(contact_animation, height=150, key="contact-anim")
        st.markdown("""
        <div class="contact-card">
            <h3 style="color: #FF4B4B; font-family: 'Montserrat', sans-serif;">👨‍💻 Jahanzaib Javed</h3>
            <p style="font-weight: bold; color: #333;">Founder @ <span style="color: #1E88E5;">Zeby Coder</span></p>
            <p style="color: #555;">📞 +92-300-5590321</p>
            <p style="color: #555;">📍 Lahore, Pakistan</p>
            <p style="font-style: italic; color: #666;">Specializing in AI/ML Solutions</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.header("⚙ Recommendation Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.1,  # Default set to 0.1 as requested
            step=0.05,
            help="Higher values mean more similar recommendations"
        )
        st.markdown("---")
        st.markdown("### 🎥 How it works")
        st.markdown("""
        1. Select a movie from the dropdown
        2. Click 'Show Recommendations'
        3. Get 5 similar movies with posters
        """)

    # Load the data
    @st.cache_resource
    def load_data():
        try:
            movies = pickle.load(open(DATA_FILES['movie_list'], 'rb'))
            similarity = pickle.load(open(DATA_FILES['similarity'], 'rb'))
            return movies, similarity
        except:
            st.warning("Processing data for the first time. This may take a few minutes...")
            movies, similarity = process_data()
            pickle.dump(movies, open(DATA_FILES['movie_list'], 'wb'))
            pickle.dump(similarity, open(DATA_FILES['similarity'], 'wb'))
            return movies, similarity

    movies, similarity = load_data()

    # Main content
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            "https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg",
            width=200)

    with col2:
        search_query = st.text_input("🔍 Search for a movie", help="Start typing to filter movies")

    # Movie selection
    if search_query:
        filtered_movies = [m for m in movies['title'].values if search_query.lower() in m.lower()]
        if not filtered_movies:
            st.warning("No movies found matching your search")
        selected_movie = st.selectbox(
            "Select a movie",
            filtered_movies if filtered_movies else movies['title'].values,
            index=0 if not filtered_movies else None
        )
    else:
        selected_movie = st.selectbox(
            "Select a movie",
            movies['title'].values
        )

    # Show recommendations
    if st.button('🎥 Show Recommendations', type="primary", key="recommend_btn"):
        with st.spinner('✨ Finding your perfect movie matches...'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie, movies, similarity,
                                                                           similarity_threshold)

        if not recommended_movie_names:
            st.warning("No sufficiently similar movies found. Try lowering the similarity threshold.")
        else:
            st.success(f"🎉 Movies similar to '{selected_movie}':")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                    st.image(
                        recommended_movie_posters[i],
                        use_column_width=True,
                        caption=recommended_movie_names[i]
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    # Additional movie details
    with st.expander("ℹ About this movie", expanded=False):
        try:
            movie_data = movies[movies['title'] == selected_movie].iloc[0]
            st.subheader(selected_movie)
            st.write(f"📖 Overview:** {movie_data['overview']}")

            genres = ast.literal_eval(movie_data['genres'])
            st.write(f"🎭 Genres:** {', '.join(genres)}")

            cast = ast.literal_eval(movie_data['cast'])
            st.write(f"🌟 Cast:** {', '.join(cast)}")

            st.write(f"🎬 Director:** {movie_data['director']}")
        except:
            st.warning("Could not load additional details for this movie")

    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin: 0; font-size: 1rem;"><b>🎞 Data Source</b>: The Movie Database (TMDB)</p>
        <p style="margin: 5px 0 0; font-size: 0.9rem;">
            <b>👨‍💻 Developed by</b>: Jahanzaib Javed | <b>🏢 Company</b>: Zeby Coder | 
            <b>📞 Contact</b>: +92-300-5590321 | <b>📍 Location</b>: Lahore, Pakistan
        </p>
        <p style="margin: 5px 0 0; font-size: 0.8rem;">© 2023 Zeby Coder - All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if _name_ == "_main_":
    main()
