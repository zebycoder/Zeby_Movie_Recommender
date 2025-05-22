import pickle
import numpy as np
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Zeby Coder's Movie Magic",
    page_icon="🎬",
    layout="wide"
)


# --- CACHING FOR MAXIMUM SPEED ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity


@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        return requests.get(url, timeout=2).json()['poster_path']
    except:
        return None


# --- OPTIMIZED UI ---
def main():
    movies, similarity = load_data()

    # --- HIGH-CONTRAST THEME ---
    st.markdown("""
    <style>
        .stApp {
            background: #0F172A;
            color: #E2E8F0;
        }
        .sidebar .sidebar-content {
            background: #1E293B !important;
        }
        .stTextInput input {
            background: #1E293B !important;
            color: white !important;
        }
        .stSelectbox select {
            background: #1E293B !important;
            color: white !important;
        }
        .stButton button {
            background: #3B82F6 !important;
            color: white !important;
            font-weight: bold;
            border: none;
            width: 100%;
        }
        .contact-card {
            background: #1E293B;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR (CLEAR VISIBILITY) ---
    with st.sidebar:
        st.markdown("""
        <div class="contact-card">
            <h3 style="color: #3B82F6;">👨‍💻 Jahanzaib Javed</h3>
            <p><b style="color: #94A3B8;">Company:</b> <span style="color: #E2E8F0;">Zeby Coder</span></p>
            <p><b style="color: #94A3B8;">Phone:</b> <span style="color: #E2E8F0;">‪+92-300-5590321‬</span></p>
            <p><b style="color: #94A3B8;">Location:</b> <span style="color: #E2E8F0;">Lahore, Pakistan</span></p>
        </div>
        """, unsafe_allow_html=True)

        threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.1, 0.05)
        st.markdown("---")
        st.caption("💡 Tip: Lower threshold = More recommendations")

    # --- MAIN CONTENT ---
    st.title("🎬 Zeby Coder Presents")
    st.subheader("AI-Powered Movie Recommendations", divider="blue")

    # Search & Selection
    search_query = st.text_input("🔍 Search Movies", placeholder="Type a movie name...")
    movie_list = movies['title'].tolist()
    if search_query:
        movie_list = [m for m in movie_list if search_query.lower() in m.lower()]

    selected_movie = st.selectbox("Select Movie", movie_list, index=0 if not search_query else None)

    # Recommendations
    if st.button("🎥 Get Recommendations", type="primary"):
        with st.spinner("Finding matches..."):
            index = movies[movies['title'] == selected_movie].index[0]
            sim_scores = list(enumerate(similarity[index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

            cols = st.columns(5)
            for i, (idx, score) in enumerate(sim_scores):
                if score < threshold:
                    continue
                poster_path = fetch_poster(movies.iloc[idx].movie_id)
                cols[i].image(
                    f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Poster",
                    caption=movies.iloc[idx].title,
                    use_column_width=True
                )


if _name_ == "_main_":
    main()
