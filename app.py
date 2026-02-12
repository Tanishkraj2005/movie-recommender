import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")


@st.cache_resource
def load_model():
    df = pickle.load(open("movies.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    return df, tfidf_matrix


df, tfidf_matrix = load_model()

movie_list = df["title"].values

selected_movie = st.selectbox("Select a movie:", movie_list)


def recommend(movie_title):
    idx = df[df["title"].str.lower() == movie_title.lower()].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    scores = list(enumerate(similarity_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    return [df.iloc[i[0]].title for i in scores]


if st.button("Recommend"):
    results = recommend(selected_movie)

    st.subheader("Top Recommendations")

    for movie in results:
        st.write(movie)
