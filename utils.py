import h5py
import scipy.misc
import numpy as np
import scipy.io


def preprocess_nyu_dataset(path_to_data='./nyu_depth_v2_labeled.mat',
                           path_to_mapping='./classMapping40.mat',
                           path_save_data='./datasets/nyu/'):
    mapping = scipy.io.loadmat(path_to_mapping)['mapClass'][0]
    idx_to_mapping_index = {i + 1: value for i, value in enumerate(mapping)}
    idx_to_mapping_index[0] = 0
    mapping_func = np.vectorize(lambda element: idx_to_mapping_index[element])
    data = h5py.File(path_to_data)
    for idx, image, segmentation in zip(range(len(data['images'])), data['images'], data['labels']):
        segmentation = mapping_func(segmentation).astype('uint8')
        if idx < 1200:
            scipy.misc.imsave('{}train/image/{:04}.png'.format(path_save_data, idx), np.transpose(image))
            scipy.misc.imsave('{}train/segm/{:04}.png'.format(path_save_data, idx), np.transpose(segmentation))
        else:
            scipy.misc.imsave('{}val/image/{:04}.png'.format(path_save_data, idx), np.transpose(image))
            scipy.misc.imsave('{}val/segm/{:04}.png'.format(path_save_data, idx), np.transpose(segmentation))


def define_random_segmentation_color(path_save_data='./datasets/nyu/', num_classes=41):
    np.random.seed(42)
    colors = np.random.randint(0, 256, (num_classes, 3))
    np.save('{}colors.npy'.format(path_save_data), colors)
