import ast

def extract_director(crew_list):
    """
    Extract director name from crew column.
    Returns a list so we can join safely later.
    """
    for member in crew_list:
        if member.get("job") == "Director":
            return [member["name"]]
    return []


def extract_top_cast(cast_list, top_n=3):
    """
    Extract top N cast members (first 3).
    """
    try:
        return [member["name"] for member in cast_list[:top_n]]
    except:
        return []


def create_combined_features(df):
    """
    Combine genres, keywords, cast, director, and overview
    into a single text feature.
    """

    combined = []

    for index, row in df.iterrows():
        genres = " ".join(row["genres"])
        keywords = " ".join(row["keywords"])
        cast = " ".join(row["cast"])
        director = " ".join(row["director"])
        overview = row["overview"] if isinstance(row["overview"], str) else ""

        text = f"{genres} {keywords} {cast} {director} {overview}"
        combined.append(text.lower())

    df["combined_features"] = combined
    return df
