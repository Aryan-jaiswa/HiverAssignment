import os
import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads the raw Customer Support on Twitter dataset.
    
    Args:
        path: Path to the raw CSV file (e.g. data/raw/twcs.csv)
        
    Returns:
        pd.DataFrame containing the raw data.
        
    Raises:
        FileNotFoundError: If the file does not exist, with instructions on how to get it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Please download 'Customer Support on Twitter' from Kaggle "
            "(https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) "
            "and place 'twcs.csv' in the 'data/raw/' directory."
        )
    
    # tweet_ids can be large integers that lose precision if cast to float when missing values exist,
    # so we read them as strings.
    dtypes = {
        'tweet_id': str,
        'author_id': str,
        'in_response_to_tweet_id': str,
        'response_tweet_id': str,
        'text': str
    }
    
    print(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path, dtype=dtypes)
    return df
