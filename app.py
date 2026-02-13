import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("Content-Based + Popularity Boosted Ranking")

@st.cache_resource
def load_model():
    df = pickle.load(open("movies.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    return df, tfidf_matrix


df, tfidf_matrix = load_model()

st.sidebar.header("Filter Options")

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 5.0)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

movie_list = df["title"].values
selected_movie = st.selectbox("Select a movie:", movie_list)

def recommend(movie_title, top_n, min_rating):
    try:
        idx = df[df["title"].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return []

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    popularity = df["popularity"].values
    pop_norm = (popularity - popularity.min()) / (popularity.max() - popularity.min())

    hybrid_score = 0.85 * similarity_scores + 0.15 * pop_norm

    scores = list(enumerate(hybrid_score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]

    recommendations = []

    for i, score in scores:
        if df.iloc[i]["vote_average"] >= min_rating:
            recommendations.append(i)
        if len(recommendations) == top_n:
            break

    return recommendations


if st.button("Recommend"):

    with st.spinner("Generating recommendations..."):
        results = recommend(selected_movie, top_n, min_rating)

    if not results:
        st.warning("No recommendations found.")
    else:
        st.subheader("Top Recommendations")

        for i in results:
            movie_data = df.iloc[i]

            with st.container():
                col1, col2 = st.columns([1, 3])

                with col2:
                    st.markdown(f"### 🎬 {movie_data['title']}")
                    st.write(f"⭐ Rating: {movie_data['vote_average']}")
                    st.write(f"📅 Release: {movie_data['release_date']}")
                    st.write(movie_data["overview"][:250] + "...")
                st.divider()


st.markdown("---")
st.markdown("### About This Project")

st.write("""
This is a hybrid movie recommendation system that combines:

• TF-IDF based content similarity  
• Cosine similarity scoring  
• Popularity-based ranking boost  

Optimized for fast cloud deployment using precomputed sparse matrices.
""")
