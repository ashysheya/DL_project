import numpy as np
import h5py


def calculate_number_classes(path_to_file='nyu_depth_v2_labeled.mat'):
    with h5py.File(path_to_file, 'r') as fin:
        num_classes = 0
        for current_label_map in fin['labels']:
            if current_label_map.max() > num_classes:
                num_classes = current_label_map.max()
    return num_classes
