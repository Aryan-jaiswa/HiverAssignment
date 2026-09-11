import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe by removing null texts and duplicate tweet IDs.
    
    Args:
        df: Raw pandas DataFrame.
        
    Returns:
        Cleaned pandas DataFrame.
    """
    # Drop rows where essential fields are completely missing
    df = df.dropna(subset=['tweet_id', 'author_id', 'text'])
    
    # Remove exact duplicate records
    df = df.drop_duplicates(subset=['tweet_id'])
    
    # Fill NA for relationship columns with empty string
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna('')
    df['response_tweet_id'] = df['response_tweet_id'].fillna('')
    
    return df

def basic_text_clean(text: str) -> str:
    """
    Performs minimal cleaning on text, preserving natural language and URLs,
    but fixing basic issues like extra whitespaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip leading/trailing whitespaces and condense multiple spaces
    text = " ".join(text.split())
    return text
