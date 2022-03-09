# This file should be in root dir as of now

import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
GLOVE_PATH = os.path.join(DATA_PATH, 'glove')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')


def read_file(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("file doesn't exist: ", path)


def read_file_in_dir(root_dir, file_name):
    path = os.path.join(root_dir, file_name)
    return read_file(path)


def write_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def write_to_file_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_file(path, data)


def log_to_file(path, log_str):
    with open(path, 'a') as f:
        f.write(log_str + '\n')


def log_to_file_in_dir(root_dir, file_name, log_str):
    path = os.path.join(root_dir, file_name)
    log_to_file(path, log_str)
