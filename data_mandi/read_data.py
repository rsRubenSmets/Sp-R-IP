import pickle
import numpy as np

path_pickle = "instances.pickle"
path_npy = "Graph_exp.npy"

with open(path_pickle,"rb") as file:
    data = pickle.load(file)

array_data = np.load(path_npy)

x=1