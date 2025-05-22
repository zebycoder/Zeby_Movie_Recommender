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

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

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

# Update your requirements.txt to:
"""
streamlit>=1.29.0
pandas>=2.2.0
numpy>=2.0.0
scikit-learn>=1.4.0
requests>=2.31.0
python-dotenv>=1.0.0
"""

# [Rest of your helper functions remain the same...]

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

def recommend(movie, _movies, _similarity, similarity_threshold=0.7):
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
    st.title('🎬 Movie Recommender System')

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

    with st.sidebar:
        st.header("Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values mean more similar recommendations"
        )
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. Select a movie from the dropdown
        2. Click 'Show Recommendations'
        3. Get 5 similar movies with posters
        """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            "https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg",
            width=200)

    with col2:
        search_query = st.text_input("Search for a movie", help="Start typing to filter movies")

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

    if st.button('🎥 Show Recommendations', type="primary"):
        with st.spinner('Finding similar movies...'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie, movies, similarity,
                                                                         similarity_threshold)

        if not recommended_movie_names:
            st.warning("No sufficiently similar movies found. Try lowering the similarity threshold.")
        else:
            st.success(f"Movies similar to '{selected_movie}':")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    st.image(
                        recommended_movie_posters[i],
                        use_column_width=True,
                        caption=recommended_movie_names[i]
                    )

    with st.expander("ℹ About this movie"):
        try:
            movie_data = movies[movies['title'] == selected_movie].iloc[0]
            st.subheader(selected_movie)
            st.write(f"*Overview:* {movie_data['overview']}")

            genres = ast.literal_eval(movie_data['genres'])
            st.write(f"*Genres:* {', '.join(genres)}")

            cast = ast.literal_eval(movie_data['cast'])
            st.write(f"*Cast:* {', '.join(cast)}")

            st.write(f"*Director:* {movie_data['director']}")
        except:
            st.warning("Could not load additional details for this movie")

    st.markdown("---")
    st.markdown("""
    *Data Source*: [The Movie Database (TMDB)](https://www.themoviedb.org/)  
    *Note*: This product uses the TMDB API but is not endorsed or certified by TMDB.
    """)

if __name__ == "__main__":
    main()
