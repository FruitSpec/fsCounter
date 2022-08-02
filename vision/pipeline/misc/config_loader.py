import yaml

def load_config(file_name):
    with open(file_name) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config

