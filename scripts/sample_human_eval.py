import os
import pandas as pd
from src.pipeline import SupportAgent

def main():
    golden_path = "data/processed/golden_set.csv"
    out_path = "data/processed/human_eval_sample.csv"
    
    if not os.path.exists(golden_path):
        print(f"Error: {golden_path} not found.")
        return
        
    print("Loading pipeline...")
    agent = SupportAgent()
    
    df = pd.read_csv(golden_path)
    # Take 50 random samples for human annotation
    sample_df = df.dropna(subset=['customer_message']).sample(n=min(50, len(df)), random_state=42)
    
    eval_records = []
    print(f"Generating responses for {len(sample_df)} cases... (This will take a moment)")
    
    for idx, row in sample_df.iterrows():
        msg = row['customer_message']
        res = agent.handle_message(msg)
        
        # We only want to evaluate generated responses, not escalations
        if res['decision'] == 'AUTO_HANDLE':
            eval_records.append({
                "customer_message": msg,
                "agent_response": res['reply'],
                "human_score_0_to_4": "" # To be filled by human
            })
            
        if len(eval_records) >= 50:
            break
            
    if not eval_records:
        print("No AUTO_HANDLE responses were generated. Please check your escalation thresholds or API key.")
        return
        
    out_df = pd.DataFrame(eval_records)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(eval_records)} responses to {out_path}.")
    print("ACTION REQUIRED: Open this file and fill in the 'human_score_0_to_4' column with integers 0-4.")

if __name__ == "__main__":
    main()
