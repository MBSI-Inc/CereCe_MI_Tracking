import yaml
import os

def load_config(config_path):
    """
    load and parse a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Dictionary containing the configuration information.
    """
    # 1. Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # 2. Read and parse YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
            # ensure top-level keys exist (optional) 
            required_keys = ['receiver_params', 'predictor_params', 'accumulator_params', 'control_params']
            for key in required_keys:
                if key not in config:
                    print(f"Warning: Key '{key}' missing in config file!")
            
            return config
            
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")