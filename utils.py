import h5py
import scipy.misc
import numpy as np
import scipy.io
import numpy as np


def get_segmentation_map(segmentation, instance_segmentation):
    unique_pairs = np.unique(np.concatenate((segmentation[None, :, :], 
                                             instance_segmentation[None, :, :])).reshape(2, -1), axis=1)
    unique_pairs_dictionary = {i: (unique_pairs[0][i], unique_pairs[1][i]) for i in range(unique_pairs.shape[1])}
    
    num_instances = len(unique_pairs_dictionary)
    
    correct_instance_segmentation = np.zeros_like(segmentation)
    
    for idx, pair in unique_pairs_dictionary.items():
        correct_instance_segmentation[(segmentation == pair[0]) & (instance_segmentation == pair[1])] = idx
        
    return correct_instance_segmentation


def preprocess_nyu_dataset(path_to_data='./nyu_depth_v2_labeled.mat',
                           path_to_mapping='./classMapping40.mat',
                           path_save_data='./datasets/nyu/'):
    mapping = scipy.io.loadmat(path_to_mapping)['mapClass'][0]
    idx_to_mapping_index = {i + 1: value for i, value in enumerate(mapping)}
    idx_to_mapping_index[0] = 0
    mapping_func = np.vectorize(lambda element: idx_to_mapping_index[element])
    data = h5py.File(path_to_data)
    max_val = 0
    for idx, image, segmentation, instance in zip(range(len(data['images'])), data['images'], 
                                                  data['labels'], data['instances']):
        
        correct_instance_segmentation = get_segmentation_map(segmentation, instance)
        segmentation = mapping_func(segmentation).astype('uint8')
        if idx < 1200:
            scipy.misc.imsave('{}train/image/{:04}.png'.format(path_save_data, idx), np.transpose(image))
            scipy.misc.imsave('{}train/segm/{:04}.png'.format(path_save_data, idx), np.transpose(segmentation))
            scipy.misc.imsave('{}train/instance/{:04}.png'.format(path_save_data, idx), 
                              np.transpose(correct_instance_segmentation))
        else:
            scipy.misc.imsave('{}val/image/{:04}.png'.format(path_save_data, idx), np.transpose(image))
            scipy.misc.imsave('{}val/segm/{:04}.png'.format(path_save_data, idx), np.transpose(segmentation))
            scipy.misc.imsave('{}val/instance/{:04}.png'.format(path_save_data, idx), 
                              np.transpose(correct_instance_segmentation))


def preprocess_nyu_dataset(path_to_data='./nyu_depth_v2_labeled.mat',
                           path_to_mapping='./classMapping40.mat',
                           path_save_data='./datasets/nyu/'):
    mapping = scipy.io.loadmat(path_to_mapping)['mapClass'][0]
    idx_to_mapping_index = {i + 1: value for i, value in enumerate(mapping)}
    idx_to_mapping_index[0] = 0
    mapping_func = np.vectorize(lambda element: idx_to_mapping_index[element])
    data = h5py.File(path_to_data)
    for idx, image, segmentation, instance in zip(range(len(data['images'])), data['images'], 
                                                  data['labels'], data['instances']):
        segmentation = mapping_func(segmentation).astype('uint8')
        if idx < 1200:
            scipy.misc.imsave('{}train/image/{:04}.png'.format(path_save_data, idx), np.transpose(image))
            scipy.misc.imsave('{}train/segm/{:04}.png'.format(path_save_data, idx), np.transpose(segmentation))
            scipy.misc.imsave('{}train/instance/{:04}.png'.format(path_save_data, idx), np.transpose(instance))
        else:
            scipy.misc.imsave('{}val/image/{:04}.png'.format(path_save_data, idx), np.transpose(image))
            scipy.misc.imsave('{}val/segm/{:04}.png'.format(path_save_data, idx), np.transpose(segmentation))
            scipy.misc.imsave('{}val/instance/{:04}.png'.format(path_save_data, idx), np.transpose(instance))


def define_random_segmentation_color(path_save_data='./datasets/nyu/', num_classes=41):
    np.random.seed(42)
    colors = np.random.randint(0, 256, (num_classes, 3))
    np.save('{}colors.npy'.format(path_save_data), colors)

    
def get_borders(instance_segmentation):
    borders = np.zeros_like(instance_segmentation)
    borders[:, 1:] = instance_segmentation[:, 1:] != instance_segmentation[:, :-1]
    borders[:, :-1] += instance_segmentation[:, 1:] != instance_segmentation[:, :-1]
    borders[1:, :] += instance_segmentation[1:, :] != instance_segmentation[:-1, :]
    borders[:-1, :] += instance_segmentation[1:, :] != instance_segmentation[:-1, :]
    borders[borders > 0] = 1.0
    return borders
