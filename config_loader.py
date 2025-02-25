import yaml

def load_config():
    try:
        with open("config.yaml", "r") as file:
            config_data = yaml.safe_load(file)
            if not config_data:
                raise ValueError("config.yaml is empty or not properly formatted.")
            return config_data
    except Exception as e:
        raise ValueError(f"Error loading config.yaml: {e}")

config = load_config() # Debugging statement
