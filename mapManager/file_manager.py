from .constants import *
import os
import pickle


def check_folder(folderName):
    if os.path.exists(folderName):
        print("Accessed folder " + folderName)
    else:
        os.makedirs(folderName)
        print("Created folder " + folderName)

def overwrite_name(name, ext, rel_path):
    i = 1
    path = os.path.join(rel_path, name + "." + ext)
    while os.path.exists(path):
        name_new = name + "({}).{}".format(i, ext)
        path = os.path.join(rel_path, name_new)
        i+=1
    return path

def save(cls, name, folder =None,overwrite = False):
    if type(folder) != type(None):
        folder_path = os.path.join(save_folder, folder)
    else:
        folder_path = save_folder
    if len(folder_path) > 0: 
        check_folder(folder_path)
    path = os.path.join(folder_path, name+".pkl")
    if overwrite != True and os.path.exists(path):
        path = overwrite_name(name, "pkl",save_folder)
    with open(path, "wb") as pickle_out:
        pickle.dump(cls, pickle_out)
    print ("Saved class instance as " +path)

def load(name, folder=None):
    if type(folder) != type(None):
        folder_path = os.path.join(save_folder, folder)
    else:
        folder_path = save_folder
    path = os.path.join(folder_path, name+".pkl")
    with open(path, "rb") as pickle_in:
        cls = pickle.load(pickle_in)
    print ("Loaded " +path)
    return cls