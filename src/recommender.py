from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(df):
    """
    Creates TF-IDF matrix and cosine similarity matrix.
    """
    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


def recommend(movie_title, df, similarity_matrix):
    """
    Returns top 10 similar movies.
    """
    movie_title = movie_title.lower()

    if movie_title not in df["title"].str.lower().values:
        return []

    idx = df[df["title"].str.lower() == movie_title].index[0]

    scores = list(enumerate(similarity_matrix[idx]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_scores = scores[1:11]

    recommended_movies = [df.iloc[i[0]].title for i in top_scores]

    return recommended_movies
