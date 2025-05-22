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

# Set page config with attractive settings
st.set_page_config(
    page_title="CineMatch AI - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
    }
    h1 {
        color: #FF4B4B;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        background-color: #1a1a2e;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a2e;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: white;
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

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

def recommend(movie, _movies, _similarity, similarity_threshold=0.1):  # Default threshold changed to 0.1
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
    # App Header with Logo and Title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg", 
                width=150)
    with col2:
        st.title('🎬 CineMatch AI')
        st.markdown("### Your Personal Movie Recommendation Engine")
    
    # About Section
    with st.expander("ℹ️ About CineMatch AI"):
        st.markdown("""
        **CineMatch AI** is an advanced movie recommendation system powered by:
        - **Machine Learning**: Uses cosine similarity to find movies with similar content
        - **Natural Language Processing**: Analyzes movie plots, genres, and keywords
        - **Collaborative Filtering**: Considers cast and director information
        
        This app is part of our **AI/ML Services** offering, providing intelligent solutions for:
        - Personalized recommendation systems
        - Content analysis and classification
        - Customer behavior prediction
        """)

    # Load data with caching
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

    # Sidebar for controls and developer info
    with st.sidebar:
        st.header("⚙️ Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.1,  # Default value set to 0.1
            step=0.05,
            help="Higher values mean more similar recommendations"
        )
        
        st.markdown("---")
        st.header("👨‍💻 Developer Info")
        st.markdown("""
        **Name**: Jahanzaib Javed  
        **Company**: Zeby AI Solutions Inc.  
        **Speciality**: AI/ML Services  
        **Contact**: +92-300-5590321  
        **Email**: zeb@innerartinteriors.com  
        **Location**: Lahore, Pakistan  
        """)
        
        st.markdown("---")
        st.markdown("### 🎥 How it works")
        st.markdown("""
        1. Search for a movie
        2. Select from dropdown
        3. Click 'Show Recommendations'
        4. Get 5 similar movies with posters
        """)

    # Main content area
    st.markdown("---")
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

    # Show recommendations button
    if st.button('🎥 Show Recommendations', type="primary"):
        with st.spinner('Finding similar movies...'):
            recommended_movie_names, recommended_movie_posters = recommend(
                selected_movie, movies, similarity, similarity_threshold)

        if not recommended_movie_names:
            st.warning("No sufficiently similar movies found. Try lowering the similarity threshold.")
        else:
            st.success(f"🎬 Movies similar to '{selected_movie}':")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    st.image(
                        recommended_movie_posters[i],
                        use_container_width=True,  # Changed from use_column_width
                        caption=recommended_movie_posters[i]
                    )

    # Movie details section - only shown if details are available
    try:
        movie_data = movies[movies['title'] == selected_movie].iloc[0]
        with st.expander("📖 Movie Details"):
            st.subheader(selected_movie)
            st.write(f"**Overview:** {movie_data['overview']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Genres:** {', '.join(ast.literal_eval(movie_data['genres']))}")
                st.write(f"**Director:** {movie_data['director']}")
            with col2:
                st.write(f"**Cast:** {', '.join(ast.literal_eval(movie_data['cast']))}")
    except:
        pass  # Silently handle cases where details aren't available

    # Footer
    st.markdown("---")
    st.markdown("""
    *Data Source*: [The Movie Database (TMDB)](https://www.themoviedb.org/)  
    *Note*: This product uses the TMDB API and endorsed or certified by TMDB.
    """)
    st.markdown("""
    <div class="footer">
        © 2024 AI Solutions Inc. | All Rights Reserved
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
