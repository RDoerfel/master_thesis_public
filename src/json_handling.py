#%%
import json
import os

#%%
def read_json(json_file):
    """
    Reads the paths from a json file.
    args:
        json_file: json file to read from
    returns: dictionary of paths
    """
    with open(json_file, 'r') as f:
        paths = json.load(f)
    return paths

def write_json(json_file, dict_to_write):
    """
    Writes the paths to a json file.
    args:
        json_file: json file to write to
        dict_to_write: dictionary to write
    """
    with open(json_file, 'w') as f:
        json.dump(dict_to_write, f, indent=4)

def add_to_dict(json_dict, key, value):
    """
    Adds a key-value pair to the paths dictionary.
    args:
        json_dict: dictionary
        key: key to add
        value: value to add
    returns: updated paths dictionary
    """
    json_dict[key] = value
    return json_dict

