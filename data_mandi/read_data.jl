using PyCall
@pyimport pickle

cd(@__DIR__)

file_path = "instances.pickle"

# Function to load a pickle file
function load_pickle(file_path)
    open(f -> pickle.load(f), file_path, "r")
end

# Example of loading a pickle file
data = load_pickle(file_path)

data["test"]