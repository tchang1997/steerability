from ruamel.yaml import YAML

def load_yaml(path):
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        return yaml.load(f)
    
def has_negprompt(cfg):
    if "inst_addons" in cfg:
        return cfg["inst_addons"].get("disambig", False)
    return False