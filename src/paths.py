from . import json_handling

class Paths:
    def __init__(self, json_file):
        """ Initializes the paths class. 
        args:
            json_file: json file to read from
        """
        self._json_file = json_file
        self._jsons = json_handling.read_json(json_file)

    def __call__(self, key):
        """ Returns the path for the given key. 
        args:
            key: key to get path for
        returns: path for the given key
        """
        return self._jsons["PATHS"][key]

    def add_path(self, key, value):
        """ Adds a path to the paths dictionary.
        args:
            key: key to add
            value: value to add
        """
        self._jsons["PATHS"][key] = value
        json_handling.write_json(self._json_file, self._jsons)
