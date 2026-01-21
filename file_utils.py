import os
import pickle
import pandas as pd


def get_program_path():
    program_path = os.path.dirname(os.path.split(__file__)[0])
    return program_path


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def check_and_rename_path(path):
    path = path.rstrip('/').rstrip('\\')
    path_tmp = path
    i = 1
    while os.path.exists(path_tmp):
        path_tmp = '%s_%d' % (path, i)
        i += 1
    if i > 1:
        os.rename(path, path_tmp)
    os.makedirs(path)


def check_and_new_path(path):
    path = path.rstrip('/').rstrip('\\')
    path_tmp = path
    i = 1
    while os.path.exists(path_tmp):
        path_tmp = '%s_%d' % (path, i)
        i += 1
    os.makedirs(path_tmp)
    return path_tmp

