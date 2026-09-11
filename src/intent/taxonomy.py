import yaml

def load_taxonomy(config_path: str = "config/intent_schema.yaml") -> list:
    """
    Loads the valid intents from the taxonomy schema.
    
    Returns:
        A list of valid intent names.
    """
    with open(config_path, "r") as f:
        schema = yaml.safe_load(f)
        
    return [intent['name'] for intent in schema['intents']]

def is_valid_intent(intent: str, config_path: str = "config/intent_schema.yaml") -> bool:
    """Checks if a given intent string is valid according to the taxonomy."""
    valid_intents = load_taxonomy(config_path)
    return intent in valid_intents
