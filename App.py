import pandas as pd
import numpy as np
import pickle
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import ast
from functools import lru_cache
import os


# ==============================================
# DATA LOADING AND PREPARATION
# ==============================================

@st.cache_data
def load_data():
    """Load or prepare movie data"""
    try:
        # Try loading preprocessed files first
        if os.path.exists('movies.pkl') and os.path.exists('similarity.npy'):
            movies = pd.read_pickle('movies.pkl')
            similarity = np.load('similarity.npy', allow_pickle=True)
            return movies, similarity

        # If no files exist, prepare data
        st.warning("Preparing data for first-time use...")
        return prepare_data()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


def prepare_data():
    """Process raw data and create recommendation files"""
    try:
        # Load raw data
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')

        # Merge and clean data
        movies = movies.merge(credits, on='title')
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        movies.dropna(inplace=True)

        # Convert stringified lists to actual lists
        def convert(obj):
            try:
                return [i['name'] for i in ast.literal_eval(obj)]
            except:
                return []

        movies['genres'] = movies['genres'].apply(convert)
        movies['keywords'] = movies['keywords'].apply(convert)
        movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
        movies['crew'] = movies['crew'].apply(
            lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'][:1])
        movies['overview'] = movies['overview'].apply(lambda x: x.split())

        # Create tags
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())
        new_df = movies[['movie_id', 'title', 'tags']]

        # Vectorization
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(new_df['tags']).toarray()
        similarity = cosine_similarity(vectors)

        # Save processed data
        new_df.to_pickle('movies.pkl')
        np.save('similarity.npy', similarity)

        return new_df, similarity

    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        st.stop()


# ==============================================
# MOVIE POSTER FETCHING
# ==============================================

@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    """Get movie poster from TMDB API"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else None
    except:
        return None


# ==============================================
# STREAMLIT APP UI
# ==============================================

def main():
    st.set_page_config(
        page_title="Zeby AI - Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Mobile-First CSS
    st.markdown("""
    <style>
        :root {
            --primary: #4e73df;
            --secondary: #224abe;
        }

        @media screen and (max-width: 768px) {
            .st-emotion-cache-1v0mbdj {
                width: 100% !important;
            }
            .recommendation-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }
            .header-section {
                padding: 1rem !important;
            }
        }

        .header-section {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 2rem;
            color: white;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }

        .recommendation-card {
            transition: all 0.3s ease;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }

        .recommendation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header-section">
        <h1>Zeby AI Movie Recommender</h1>
        <p>Advanced AI-Powered Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    movies, similarity = load_data()

    # Main layout
    col1, col2 = st.columns([1, 3], gap="medium")

    with col1:
        # Profile Section
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h3>Jahanzaib Javed</h3>
            <p><b>📱 Phone:</b> +92-300-5590321</p>
            <p><b>✉️ Email:</b> zeb.javed@outlook.com</p>
        </div>
        """, unsafe_allow_html=True)

        # Settings
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.1,
            step=0.05
        )

    with col2:
        # Search Section
        search_query = st.text_input("Search movies...", placeholder="Type a movie title")
        movie_list = movies['title'].tolist()
        if search_query:
            movie_list = [m for m in movie_list if search_query.lower() in m.lower()]

        selected_movie = st.selectbox("Select a movie", movie_list if movie_list else ["No movies found"])

        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    index = movies[movies['title'] == selected_movie].index[0]
                    sim_scores = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:6]

                    st.markdown(f"""
                    <div style="margin: 1.5rem 0;">
                        <h3>🎯 Similar to: {selected_movie}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Responsive recommendations
                    st.markdown(
                        '<div class="recommendation-grid" style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.5rem;">',
                        unsafe_allow_html=True)

                    cols = st.columns(5)
                    for i, (idx, score) in enumerate(sim_scores):
                        if score < threshold:
                            continue

                        movie = movies.iloc[idx]
                        poster_url = fetch_poster(
                            movie.movie_id) or "https://via.placeholder.com/300x450?text=No+Poster"

                        with cols[i % 5]:
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <img src="{poster_url}" style="width:100%; border-radius:8px 8px 0 0;">
                                <div style="padding: 1rem;">
                                    <h4 style="margin: 0 0 0.5rem 0; text-align: center;">{movie.title}</h4>
                                    <p style="margin: 0; text-align: center; color: var(--primary); font-weight: bold;">Similarity: {score:.2f}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ==============================================
# APP ENTRY POINT
# ==============================================
if __name__ == "__main__":
    main()
