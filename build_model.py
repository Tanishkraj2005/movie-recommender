import ast
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_data
from src.data_cleaner import clean_json_column
from src.feature_engineering import (
    extract_director,
    extract_top_cast,
    create_combined_features,
)

print("Loading data...")

movies, credits = load_data()

df = movies.merge(credits, left_on="id", right_on="movie_id")

if "title_x" in df.columns:
    df.rename(columns={"title_x": "title"}, inplace=True)
if "title_y" in df.columns:
    df.drop(columns=["title_y"], inplace=True)

df = clean_json_column(df, "genres")
df = clean_json_column(df, "keywords")

df["cast"] = df["cast"].apply(ast.literal_eval)
df["crew"] = df["crew"].apply(ast.literal_eval)

df["cast"] = df["cast"].apply(extract_top_cast)
df["director"] = df["crew"].apply(extract_director)

df["overview"] = df["overview"].fillna("")

df = create_combined_features(df)

print("Building TF-IDF matrix...")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])

print("Saving files...")

with open("movies.pkl", "wb") as f:
    pickle.dump(df, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("Done.")
