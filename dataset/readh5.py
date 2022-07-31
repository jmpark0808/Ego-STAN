import h5py
filename = "/mnt/hdd/Datasets/index_train.h5"
import numpy as np

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data_static = np.array(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list


filename = '/home/eddie/h36m/index_train.h5'
with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = np.array(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    
    data_seq = data[:, -1]

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
for subject in subjects:
    S1s = [x for x in data_static if subject in str(x)]
    S1seq = [x for x in data_seq if subject in str(x)]
    print(len(S1s), len(S1seq))
        