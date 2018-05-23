import scipy.misc
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob


class SegmentationTransform(object):
    def __init__(self, train_trainsforms=True, num_classes=3):
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
            segmentation = self.resize(transforms.transforms.ToPILImage()(segmentation))

            # mirroring
            if np.random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                segmentation = segmentation.transpose(Image.FLIP_LEFT_RIGHT)

            # random crop
            x_offset = np.random.randint(0, 21)
            y_offset = np.random.randint(0, 21)

            image = image.crop((x_offset, y_offset, x_offset + 256, y_offset + 256))
            segmentation = segmentation.crop((x_offset, y_offset, x_offset + 256, y_offset + 256))

        else:
            image = self.resize(transforms.transforms.ToPILImage()(image))
            segmentation = self.resize(transforms.transforms.ToPILImage()(segmentation))

        image_tensor, segmentation_tensor = self.to_tensor(image), self.to_tensor(segmentation)
        image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_tensor)
        segmentation_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(segmentation_tensor)

        return image_tensor, segmentation_tensor



class SegmentationDataset(Dataset):
    def __init__(self, path_to_datafolder='./datasets/facades/train/', transforms=None):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = None

        self.mixed_images_filenames = sorted(glob.glob('{}*.jpg'.format(path_to_datafolder)))

    def __len__(self):
        return len(self.mixed_images_filenames)

    def __getitem__(self, idx):
        mixed_image = scipy.misc.imread(self.mixed_images_filenames[idx])
        size = mixed_image.shape[0]
        image        = mixed_image[:, :size]
        segmentation = mixed_image[:, size:]

        if self.transforms:
            image, segmentation = self.transforms(image, segmentation)
        return image, segmentation
