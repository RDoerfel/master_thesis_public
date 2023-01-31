from src import json_handling
import json
import pytest
import os

TEST_FILE = 'test_json.json'

def setup_module():
    test_paths = {'PATHS':{'Test1':1, 'Test2':2}}
    with open(TEST_FILE, 'w') as f:
        json.dump(test_paths, f, indent=4)

def teardown_module():
    os.remove(TEST_FILE)

def test_read_json():
    json = json_handling.read_json(TEST_FILE)
    assert json['PATHS']['Test1'] == 1
    assert json['PATHS']['Test2'] == 2

def test_write_json():
    json_handling.write_json(TEST_FILE, {'Test3':3})
    json = json_handling.read_json(TEST_FILE)
    assert json['Test3'] == 3

def test_add_to_dict():
    test_dict = {'Test1':1, 'Test2':2}    
    json_handling.add_to_dict(test_dict, 'Test4', 4)
    assert test_dict['Test1'] == 1
    assert test_dict['Test2'] == 2
    assert test_dict['Test4'] == 4
