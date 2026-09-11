import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_raw_data
from src.data.clean import clean_dataframe, basic_text_clean
from src.data.conversations import reconstruct_conversations

def generate_eda(df: pd.DataFrame, report_path: str):
    """Generates basic EDA report for the raw dataset."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=== RAW DATASET EDA ===\n")
        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Missing Values:\n{df.isna().sum()}\n")
        
        f.write("\n=== INBOUND/OUTBOUND DISTRIBUTION ===\n")
        f.write(f"{df['inbound'].value_counts(normalize=True).to_string()}\n")
        
        f.write("\n=== TOP 10 BRANDS (OUTBOUND) ===\n")
        outbound = df[df['inbound'] == False]
        f.write(f"{outbound['author_id'].value_counts().head(10).to_string()}\n")

def main():
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)['data']
        
    raw_path = config['raw_path']
    processed_dir = config['processed_dir']
    brand = config['brand']
    seed = config['split_seed']
    train_ratio = config['train_ratio']
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load Data
    try:
        df = load_raw_data(raw_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. EDA
    print("Generating EDA...")
    generate_eda(df, "reports/eda_summary.txt")
    
    # 3. Clean
    print("Cleaning raw data...")
    df = clean_dataframe(df)
    
    # 4. Reconstruct Conversations
    print(f"Reconstructing conversations for {brand}...")
    conv_df, stats = reconstruct_conversations(df, brand)
    
    # Clean text
    print("Cleaning text...")
    conv_df['customer_message'] = conv_df['customer_message'].apply(basic_text_clean)
    conv_df['support_reply'] = conv_df['support_reply'].apply(basic_text_clean)
    
    # Drop empty messages
    conv_df = conv_df[(conv_df['customer_message'] != '') & (conv_df['support_reply'] != '')]
    
    # 5. Split
    print("Splitting dataset...")
    # First split train vs (val + test)
    train_df, temp_df = train_test_split(
        conv_df, 
        test_size=(val_ratio + test_ratio), 
        random_state=seed
    )
    
    # Then split val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=seed
    )
    
    # 6. Save
    print("Saving processed files...")
    full_path = os.path.join(processed_dir, f"{brand.lower()}_conversations.csv")
    train_path = os.path.join(processed_dir, f"{brand.lower()}_train.csv")
    val_path = os.path.join(processed_dir, f"{brand.lower()}_val.csv")
    test_path = os.path.join(processed_dir, f"{brand.lower()}_test.csv")
    
    conv_df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # 7. Print Output
    print("\n=== PIPELINE SUMMARY ===")
    print(f"Raw rows: {stats['raw_rows']}")
    print(f"AppleSupport rows: {stats['brand_support_tweets']}")
    print(f"Reconstructed conversations: {stats['reconstructed_conversations']}")
    print(f"Valid customer-support pairs: {stats['valid_customer_support_pairs']}")
    print(f"Removed records: {stats['removed_records']}")
    print(f"Train conversations: {len(train_df)}")
    print(f"Validation conversations: {len(val_df)}")
    print(f"Test conversations: {len(test_df)}")
    print("========================\n")

if __name__ == "__main__":
    main()
