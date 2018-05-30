import scipy.misc
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
from utils import get_borders


class SegmentationTransform(object):
    def __init__(self, train_trainsforms=True, num_classes=41, use_rgb=False, path_to_rgb=None, resize=True):
        self.train_trainsforms = train_trainsforms
        self.to_tensor = transforms.ToTensor()
        
        if resize:
            if train_trainsforms:
                self.resize = transforms.Resize(size=(286, 286))
            else:
                self.resize = transforms.Resize(size=(256, 256))
            self.num_classes = num_classes
            
        self.use_resize = resize
        self.num_classes = num_classes

        self.use_rgb = use_rgb
        if use_rgb:
            self.colors = np.load(path_to_rgb)

    def __call__(self, image, segmentation, instance_segmentation=None, depth=None):

        if self.train_trainsforms:
            
            if self.use_resize:
                image = self.resize(transforms.transforms.ToPILImage()(image))
                segmentation = scipy.misc.imresize(segmentation, (286, 286), 'nearest')
                if instance_segmentation is not None:
                    instance_segmentation = scipy.misc.imresize(instance_segmentation, 
                                                                (286, 286), 'nearest')
                if depth is not None:
                    depth = scipy.misc.imresize(depth, (286, 286), 'bilinear')
                    
            else:
                image = transforms.transforms.ToPILImage()(image)          

            # mirroring
            if np.random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                segmentation = np.flip(segmentation, 1)
                if instance_segmentation is not None:
                    instance_segmentation = np.flip(instance_segmentation, 1)
                if depth is not None:
                    depth = np.flip(depth, 1)
            
            if self.use_resize:
                # random crop
                x_offset = np.random.randint(0, 21)
                y_offset = np.random.randint(0, 21)

                image = image.crop((x_offset, y_offset, x_offset + 256, y_offset + 256))
                segmentation = segmentation[x_offset: x_offset + 256, y_offset: y_offset + 256]
                
                if instance_segmentation is not None:
                    instance_segmentation = instance_segmentation[x_offset: x_offset + 256,
                                                                  y_offset: y_offset + 256]
                if depth is not None:
                    depth = depth[x_offset: x_offset + 256, y_offset: y_offset + 256]

        else:
            
            if self.use_resize:
                image = self.resize(transforms.transforms.ToPILImage()(image))
                segmentation = scipy.misc.imresize(segmentation, (256, 256), 'nearest')
                if instance_segmentation is not None:
                    instance_segmentation = scipy.misc.imresize(instance_segmentation, 
                                                                (256, 256), 'nearest')
                if depth is not None:
                    depth = scipy.misc.imresize(depth, (256, 256), 'bilinear')
            else:
                image = transforms.transforms.ToPILImage()(image)

        image_tensor = self.to_tensor(image)
        image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_tensor)
        if self.use_rgb:
            segmentation_tensor = self.colors[segmentation]
            segmentation_tensor = self.to_tensor(segmentation_tensor)
            segmentation_tensor = transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))(segmentation_tensor)
        else:
            segmentation_tensor = (np.arange(self.num_classes) == segmentation[..., None]).astype(int)
            segmentation_tensor = np.rollaxis(segmentation_tensor, -1, 0)
        segmentation_tensor = torch.FloatTensor(segmentation_tensor)
        
        if instance_segmentation is not None:
            borders = get_borders(instance_segmentation)
            borders_tensor = torch.FloatTensor(borders).view(1, borders.shape[0], borders.shape[1])
            instance_segmentation_tensor = torch.LongTensor(instance_segmentation.copy()).view(1, 
                                                                                             borders.shape[0], 
                                                                                             borders.shape[1])
            return image_tensor, segmentation_tensor, instance_segmentation_tensor, borders_tensor
        
        if depth is not None:
            depth_tensor = torch.FloatTensor(depth.copy()).view(1, depth.shape[0], depth.shape[1])
            depth_tensor_reshaped = torch.FloatTensor(scipy.misc.imresize(depth, (128,128), 'bilinear')).view(1, 128, 128)
            return image_tensor, segmentation_tensor, depth_tensor, depth_tensor_reshaped
            
        return image_tensor, segmentation_tensor



class SegmentationDataset(Dataset):
    def __init__(self, path_to_datafolder='./datasets/nyu/train/', transforms=None, use_instance_segmentation=False, use_depth=True):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = None
        
        self.use_instance_segmentation = use_instance_segmentation
        self.use_depth = use_depth
        
        self.image_filenames = sorted(glob.glob('{}image/*.png'.format(path_to_datafolder)))
        self.segm_filenames = sorted(glob.glob('{}segm/*.png'.format(path_to_datafolder)))
        
        if use_instance_segmentation:
            self.instance_segm_filenames = sorted(glob.glob('{}instance/*.png'.format(path_to_datafolder)))
            
        if self.use_depth:
            self.depth_filenames = sorted(glob.glob('{}depth/*.npy'.format(path_to_datafolder)))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = scipy.misc.imread(self.image_filenames[idx])
        segmentation = scipy.misc.imread(self.segm_filenames[idx])
        if self.transforms:
            if self.use_instance_segmentation:
                instance_segmentation = scipy.misc.imread(self.instance_segm_filenames[idx])
                image, segmentation, instance_segmentation, border = self.transforms(image, segmentation, 
                                                                                     instance_segmentation)
                return image, segmentation, instance_segmentation, border
            
            if self.use_depth:
                depth = np.load(self.depth_filenames[idx])
                image, segmentation, depth, depth_reshaped = self.transforms(image, segmentation, depth=depth)                
                return image, segmentation, depth, depth_reshaped
            
            image, segmentation = self.transforms(image, segmentation)
        return image, segmentation
