import torch
import torch.nn as nn
from pix2pix_model import Discriminator, ListModule
import numpy as np


class MultiScaleDiscriminator(nn.Module):
    """
    Class of the whole Discriminating system in pix2pixHD -
    three Discriminators working on different scales.
    """
    def __init__(self, in_channels, num_discriminators=3, instance_norm=True,
                 get_all_features=True):

        super(MultiScaleDiscriminator, self).__init__()

        discriminators = [Discriminator(in_channels, instance_norm=instance_norm,
                                        get_all_features=get_all_features) for _ in
                          range(num_discriminators)]
        self._discriminators = ListModule(*discriminators)

        self._downsampling = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def get_discriminator(self, idx):
        if idx < len(self._discriminators):
            return self._discriminators[idx]
        else:
            raise IndexError('index {} is out of range'.format(idx))
        
    def forward(self, x):
        outputs = []
        downsampled_input = x

        for i, discriminator in enumerate(self._discriminators):
            outputs.append(discriminator.forward(downsampled_input))
            if i != len(self._discriminators) - 1:
                downsampled_input = self._downsampling(downsampled_input)
        return outputs

    
class FeatureEncoder(nn.Module):
    """
    Class for extracting instance-wise feature vectors.
    """
    def __init__(self, in_channels, out_channels, instance_norm=False):
        super(FeatureEncoder, self).__init__()
        self._out_channels = out_channels

        if instance_norm:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        
        list_in_channels = [in_channels, 32, 64, 128, 256, 128, 64, 32]
        list_out_channels = list_in_channels[1:] + [out_channels]
        depth = len(list_out_channels)
        
        modules = []
        for i, in_ch, out_ch in zip(range(depth), list_in_channels, list_out_channels):
            if i == 0:
                modules += [nn.ReflectionPad2d(3),
                            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                            norm_layer(out_ch), nn.ReLU(True)]
            elif i == depth-1:
                modules += [nn.ReflectionPad2d(3),
                            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0), nn.Tanh()]
            elif in_ch < out_ch:
                modules += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                            norm_layer(out_ch), nn.ReLU(True)]
            elif in_ch > out_ch:
                modules += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                            norm_layer(out_ch), nn.ReLU(True)]

        self._net = nn.Sequential(*modules)
        
    def forward(self, input, inst):
        outputs = self._net.forward(input)
        
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            indices = (inst == i).nonzero() 
            for j in range(self._out_channels):
                output_inst = outputs[indices[:, 0], indices[:, 1] + j, indices[:, 2],
                                      indices[:, 3]]
                mean_features = torch.mean(output_inst).expand_as(output_inst)
                outputs_mean[indices[:, 0], indices[:, 1] + j,
                             indices[:, 2], indices[:, 3]] = mean_features
        return outputs_mean
