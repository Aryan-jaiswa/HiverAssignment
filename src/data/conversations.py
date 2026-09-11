import pandas as pd
from typing import Tuple, Dict

def reconstruct_conversations(df: pd.DataFrame, brand: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Reconstructs customer -> support conversations for a specific brand.
    
    Args:
        df: Cleaned full dataset DataFrame.
        brand: The brand author_id (e.g., 'AppleSupport').
        
    Returns:
        A tuple of (conversations_df, stats_dict).
    """
    stats = {
        "raw_rows": len(df)
    }
    
    # 1. Get outbound support tweets for the brand
    support_tweets = df[(df['author_id'] == brand) & (df['inbound'] == False)]
    stats["brand_support_tweets"] = len(support_tweets)
    
    # 2. Extract parent tweet IDs
    parent_ids = support_tweets['in_response_to_tweet_id'].unique()
    
    # 3. Get customer tweets
    # A valid customer tweet must be in the parent_ids and NOT authored by the brand itself
    customer_tweets = df[(df['tweet_id'].isin(parent_ids)) & (df['author_id'] != brand)]
    
    # 4. Merge them
    # customer_tweets has tweet_id. support_tweets has in_response_to_tweet_id.
    conversations = pd.merge(
        customer_tweets,
        support_tweets,
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_support')
    )
    
    stats["reconstructed_conversations"] = len(conversations)
    
    # Keep only the first reply from support for each customer message if there are duplicates
    conversations = conversations.drop_duplicates(subset=['tweet_id_customer'], keep='first')
    
    stats["valid_customer_support_pairs"] = len(conversations)
    
    # Calculate removed records (support tweets that didn't have a valid customer parent)
    stats["removed_records"] = len(support_tweets) - stats["valid_customer_support_pairs"]
    
    # Select and rename final columns
    final_df = pd.DataFrame({
        'conversation_id': conversations['tweet_id_customer'],
        'customer_tweet_id': conversations['tweet_id_customer'],
        'support_tweet_id': conversations['tweet_id_support'],
        'created_at': conversations['created_at_customer'],
        'customer_message': conversations['text_customer'],
        'support_reply': conversations['text_support']
    })
    
    return final_df, stats
