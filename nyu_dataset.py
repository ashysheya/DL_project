import scipy.misc
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob


class SegmentationTransform(object):
    def __init__(self, train_trainsforms=True, num_classes=41):
        self.train_trainsforms = train_trainsforms
        self.to_tensor = transforms.ToTensor()
        if train_trainsforms:
            self.resize = transforms.Resize(size=(286, 286))
        else:
            self.resize = transforms.Resize(size=(256, 256))
        self.num_classes = num_classes

    def __call__(self, image, segmentation):

        if self.train_trainsforms:
            image = self.resize(transforms.transforms.ToPILImage()(image))
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
            image = self.resize(transforms.transforms.ToPILImage()(image))
            segmentation = scipy.misc.imresize(segmentation, (256, 256), 'nearest')

        image_tensor = self.to_tensor(image)
        image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_tensor)
        segmentation_tensor = (np.arange(self.num_classes) == segmentation[..., None]).astype(int)
        segmentation_tensor = np.rollaxis(segmentation_tensor, -1, 0)
        segmentation_tensor = torch.FloatTensor(segmentation_tensor)
        return image_tensor, segmentation_tensor


class SegmentationDataset(Dataset):
    def __init__(self, path_to_datafolder='./datasets/nyu/train/', transforms=None):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = None

        self.image_filenames = sorted(glob.glob('{}image/*.png'.format(path_to_datafolder)))
        self.segm_filenames = sorted(glob.glob('{}segm/*.png'.format(path_to_datafolder)))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = scipy.misc.imread(self.image_filenames[idx])
        segmentation = scipy.misc.imread(self.segm_filenames[idx])
        if self.transforms:
            image, segmentation = self.transforms(image, segmentation)
        return image, segmentation
