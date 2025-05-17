import pickle
import numpy as np
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from PIL import Image
import io
import base64

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Zeby AI - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- CACHING FOR PERFORMANCE ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity


@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url, timeout=2).json()
        return data.get('poster_path')
    except:
        return None


# --- AI BRANDING ELEMENTS ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# --- MAIN APP ---
def main():
    movies, similarity = load_data()

    # --- LIGHT THEME WITH PROFESSIONAL STYLING ---
    st.markdown(f"""
    <style>
        .stApp {{
            background: #f8f9fa;
            color: #212529;
        }}
        .sidebar .sidebar-content {{
            background: #ffffff !important;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .stTextInput input, .stSelectbox select {{
            background: #ffffff !important;
            color: #495057 !important;
            border: 1px solid #ced4da !important;
        }}
        .stButton button {{
            background: #4e73df !important;
            color: white !important;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s;
        }}
        .stButton button:hover {{
            background: #2e59d9 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(78, 115, 223, 0.3);
        }}
        .header-section {{
            background: linear-gradient(135deg, #4e73df 0%, #224abe 100%);
            padding: 2rem;
            border-radius: 0 0 15px 15px;
            margin-bottom: 2rem;
            color: white;
        }}
        .feature-card {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border-left: 4px solid #4e73df;
            transition: all 0.3s;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        }}
        .contact-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- PROFESSIONAL HEADER WITH LOGO ---
    st.markdown("""
    <div class="header-section">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="color: white; margin-bottom: 0.5rem;">Zeby AI Solutions</h1>
                <h3 style="color: rgba(255,255,255,0.9); font-weight: 400;">Advanced Movie Recommendation System</h3>
            </div>
            <div style="font-size: 2.5rem;">🎬</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- AI FEATURES SECTION ---
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem;">
        <div class="feature-card">
            <h4>🤖 AI-Powered</h4>
            <p>Uses cosine similarity and NLP to find perfect matches</p>
        </div>
        <div class="feature-card">
            <h4>🔍 Smart Search</h4>
            <p>Real-time filtering of 5000+ movies</p>
        </div>
        <div class="feature-card">
            <h4>🎯 Personalized</h4>
            <p>Adapts to your preferences with similarity tuning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- MAIN CONTENT LAYOUT ---
    col1, col2 = st.columns([1, 3])

    with col1:
        # --- SIDEBAR CONTENT ---
        st.markdown("""
        <div class="contact-card">
            <h3 style="color: #4e73df; margin-top: 0;">Jahanzaib Javed</h3>
            <p><b>Company:</b> Zeby Coder</p>
            <p><b>Contact:</b> +92-300-5590321</p>
            <p><b>Location:</b> Lahore, Pakistan</p>
            <p><b>Specialization:</b> AI/ML Solutions</p>
        </div>
        """, unsafe_allow_html=True)

        threshold = st.slider("🔧 Similarity Threshold", 0.1, 1.0, 0.1, 0.05,
                              help="Higher values = More similar recommendations")

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <small>Powered by</small>
            <h4 style="margin-top: 0;">Zeby AI Engine</h4>
            <p style="font-size: 0.8rem;">© 2023 All Rights Reserved</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # --- SEARCH & RECOMMENDATIONS ---
        search_query = st.text_input("🔍 Search Movies", placeholder="Type a movie name...")
        movie_list = movies['title'].tolist()
        if search_query:
            movie_list = [m for m in movie_list if search_query.lower() in m.lower()]

        selected_movie = st.selectbox("🎥 Select Movie", movie_list, index=0 if not search_query else None)

        if st.button("✨ Get AI Recommendations", type="primary"):
            with st.spinner("Analyzing 5000+ movies..."):
                index = movies[movies['title'] == selected_movie].index[0]
                sim_scores = list(enumerate(similarity[index]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

                st.success(f"🎯 Recommendations similar to: **{selected_movie}**")
                cols = st.columns(5)
                for i, (idx, score) in enumerate(sim_scores):
                    if score < threshold:
                        continue
                    poster_path = fetch_poster(movies.iloc[idx].movie_id)
                    with cols[i]:
                        st.image(
                            f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Poster",
                            caption=movies.iloc[idx].title,
                            use_column_width=True
                        )
                        st.progress(float(score))
                        st.caption(f"Similarity: {score:.2f}")


if __name__ == "__main__":
    main()