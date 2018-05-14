import scipy.misc
import numpy as np
import h5py
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset

VAL_SIZE = 249
TRAIN_SIZE = 1200


class SegmentationTransform(object):
    def __init__(self, train_trainsforms=True, num_classes=894):
        self.train_trainsforms = train_trainsforms
        self.to_tensor = transforms.ToTensor()
        if train_trainsforms:
            self.resize = transforms.Resize(size=(286, 286))
        else:
            self.resize = transforms.Resize(size=(256, 256))
        self.num_classes = num_classes

    def __call__(self, input_image, input_segmentation):

        image = transforms.ToPILImage()(np.transpose(input_image))
        segmentation = np.transpose(input_segmentation)

        if self.train_trainsforms:
            image = self.resize(image)
            segmentation = scipy.misc.imresize(segmentation, (286, 286), 'nearest')

            # mirroring
            if np.random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                segmentation = np.flip(segmentation, 1)

            # random crop
            x_offset = np.random.randint(0, 21)
            y_offset = np.random.randint(0, 21)

            image = image.crop((x_offset, y_offset, x_offset + 256, y_offset + 256))
            segmentation = segmentation[x_offset: x_offset + 256, y_offset: y_offset + 256]

        else:
            image = self.resize(image)
            segmentation = scipy.misc.imresize(segmentation, (256, 256), 'nearest')

        image_tensor = self.to_tensor(image)
        segmentation_tensor = (np.arange(self.num_classes) == segmentation[..., None] - 1).astype(int)
        segmentation_tensor = np.rollaxis(segmentation_tensor, -1, 0)
        segmentation_tensor = torch.FloatTensor(segmentation_tensor)
        return image_tensor, segmentation_tensor


class SegmentationDataset(Dataset):
    def __init__(self, path_to_file='nyu_depth_v2_labeled.mat', transforms=None, val=False):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = None

        self.val = val

        # leave 249 for validation, other samples for training
        if val:
            self.size = VAL_SIZE
        else:
            self.size = TRAIN_SIZE

        self.data = h5py.File(path_to_file, 'r')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.val:
            idx += TRAIN_SIZE
        if self.transforms:
            im, mask = self.transforms(self.data['images'][idx], self.data['labels'][idx])
            return im, mask
        else:
            return self.data['images'][idx], self.data['labels'][idx]
