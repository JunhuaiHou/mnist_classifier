import h5py

file_path = 'network_for_mnist.h5'

try:
    with h5py.File(file_path, 'r') as file:
        # List all groups
        print("Keys: %s" % file.keys())
        a_group_key = list(file.keys())[0]

        # Get the object type by key
        print("Object type: %s" % type(file[a_group_key]))

        print("The file can be successfully opened. It's likely not corrupted.")
except IOError:
    print("Error: The file could not be opened. It might be corrupted or not a valid HDF5 file.")
except Exception as e:
    print("An unexpected error occurred:", e)
