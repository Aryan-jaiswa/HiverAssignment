import os
import yaml
import pandas as pd

def sample_golden_set(test_path: str, output_path: str, sample_size: int = 200, seed: int = 42):
    """
    Samples from the test set to create an empty golden set for manual annotation.
    """
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found. Please run Phase 1 first.")
        return
        
    df = pd.read_csv(test_path)
    
    if len(df) < sample_size:
        print(f"Warning: Test set has only {len(df)} rows. Sampling all of them.")
        sample = df.copy()
    else:
        sample = df.sample(n=sample_size, random_state=seed)
        
    # Prepare columns for the golden set
    golden_df = pd.DataFrame({
        'id': sample['conversation_id'],
        'customer_message': sample['customer_message'],
        'support_reply': sample['support_reply'], # Included for context during annotation
        'intent': '',              # To be filled manually
        'should_escalate': '',     # To be filled manually (True/False)
        'difficulty': '',          # To be filled manually (e.g., Easy, Medium, Hard)
        'annotation_notes': ''     # Any notes from annotator
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    golden_df.to_csv(output_path, index=False)
    print(f"Successfully sampled {len(golden_df)} examples to {output_path}.")
    print("Please manually annotate the 'intent', 'should_escalate', and 'difficulty' columns in this CSV.")

def main():
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)['data']
            
        test_path = os.path.join(config['processed_dir'], f"{config['brand'].lower()}_test.csv")
        output_path = os.path.join(config['processed_dir'], "golden_set.csv")
        
        sample_golden_set(test_path, output_path, sample_size=200, seed=config['split_seed'])
        
    except Exception as e:
        print(f"Error during golden set sampling: {e}")

if __name__ == "__main__":
    main()
