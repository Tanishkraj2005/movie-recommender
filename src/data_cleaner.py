import ast

def clean_json_column(df, column_name):
    """
    Converts JSON-like string columns (genres, keywords)
    into a list of names.
    """
    cleaned_data = []
    
    for item in df[column_name]:
        try:
            data = ast.literal_eval(item)
            names = [entry["name"] for entry in data]
            cleaned_data.append(names)
        except:
            cleaned_data.append([])

    df[column_name] = cleaned_data
    return df
