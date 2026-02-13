import re


def extract_director(crew_list):
    
    """Extract director name from crew column.
    Returns a list for safe joining."""

    for member in crew_list:
        if member.get("job") == "Director":
            return [member["name"]]
    return []


def extract_top_cast(cast_list, top_n=3):
    """
    Extract top N cast members (default = 3).
    """
    try:
        return [member["name"] for member in cast_list[:top_n]]
    except Exception:
        return []


def clean_text(text):
    """
    Basic text normalization:
    - Remove special characters
    - Convert to lowercase
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.lower()


def create_combined_features(df):
    """
    Create weighted feature representation:
    - Genres (normal weight)
    - Keywords (normal weight)
    - Cast (slightly boosted)
    - Director (heavily boosted)
    - Overview (context)
    """

    combined = []

    for _, row in df.iterrows():

        genres = " ".join(row["genres"])
        keywords = " ".join(row["keywords"])
        cast = " ".join(row["cast"])
        director = " ".join(row["director"])
        overview = row["overview"] if isinstance(row["overview"], str) else ""

        weighted_text = (
            f"{genres} "
            f"{keywords} "
            f"{cast} {cast} "           
            f"{director} {director} {director} "  
            f"{overview}"
        )

        combined.append(clean_text(weighted_text))

    df["combined_features"] = combined
    return df
