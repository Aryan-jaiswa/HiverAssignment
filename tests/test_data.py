import pytest
import pandas as pd
import numpy as np
import os
from src.data.clean import clean_dataframe, basic_text_clean
from src.data.conversations import reconstruct_conversations

@pytest.fixture
def sample_raw_data():
    """Mock dataset with mixed scenarios."""
    return pd.DataFrame({
        'tweet_id': ['1', '2', '3', '4', '5', '6'],
        'author_id': ['CustA', 'AppleSupport', 'CustB', 'AppleSupport', 'AppleSupport', 'CustC'],
        'inbound': [True, False, True, False, False, True],
        'created_at': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'],
        'text': [
            'Help me Apple', 
            'We can help!', 
            'Broken screen', 
            'DM us.', 
            'Follow up reply', 
            None # Null text
        ],
        'response_tweet_id': ['2', '', '4', '5', '', ''],
        'in_response_to_tweet_id': ['', '1', '', '3', '4', '']
    })

def test_clean_dataframe(sample_raw_data):
    df = clean_dataframe(sample_raw_data)
    # Should drop row 6 (null text)
    assert len(df) == 5
    assert '6' not in df['tweet_id'].values

def test_basic_text_clean():
    assert basic_text_clean("  Hello   world  ") == "Hello world"
    assert basic_text_clean(None) == ""
    assert basic_text_clean("https://t.co/abc xyz") == "https://t.co/abc xyz"

def test_reconstruct_conversations(sample_raw_data):
    df = clean_dataframe(sample_raw_data)
    conv_df, stats = reconstruct_conversations(df, 'AppleSupport')
    
    # Conversations should be (1->2), (3->4)
    # 5 is a reply to 4 (brand to brand), should not be a new conversation
    assert len(conv_df) == 2
    
    assert list(conv_df['customer_tweet_id']) == ['1', '3']
    assert list(conv_df['support_tweet_id']) == ['2', '4']
    
    # Check stats
    assert stats['valid_customer_support_pairs'] == 2

def test_train_test_leakage():
    """Verify that splitting doesn't leak conversation IDs."""
    from scripts.build_dataset import train_test_split
    
    # Generate 100 fake conversations
    conv_df = pd.DataFrame({
        'conversation_id': [str(i) for i in range(100)],
        'customer_tweet_id': [str(i) for i in range(100)],
        'support_tweet_id': [f"s{i}" for i in range(100)],
        'created_at': ['2023-01-01'] * 100,
        'customer_message': ['Help'] * 100,
        'support_reply': ['OK'] * 100
    })
    
    train_df, test_df = train_test_split(conv_df, test_size=0.2, random_state=42)
    
    train_ids = set(train_df['conversation_id'])
    test_ids = set(test_df['conversation_id'])
    
    # Intersection should be empty
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(train_df) == 80
    assert len(test_df) == 20
