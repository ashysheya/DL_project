import torch
import torch.nn as nn
from pix2pix_model import ListModule
import numpy as np


def initialize_weights(network):
    for module in network.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(0.0, 0.02)
            if module.bias is not None:
                    module.bias.data.fill_(0.0)

        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

        elif isinstance(module, nn.ConvTranspose2d):
            module.weight.data.normal_(0.0, 0.02)
            if module.bias is not None:
                    module.bias.data.fill_(0.0)
                    
                    
class Discriminator(nn.Module):
    """
    Discriminator class for pix2pixHD model
    70x70 discriminator is used.
    """
    def __init__(self, in_channels, instance_norm=True, get_all_features=False,
                 sigmoid=True):
        super(Discriminator, self).__init__()

        self._get_all_features = get_all_features

        list_in_channels = [in_channels, 64, 128, 256, 512]
        list_out_channels = [64, 128, 256, 512, 1]
        depth = len(list_out_channels)
        
        if instance_norm:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        modules = []

        for i, in_channels, out_channels in zip(range(depth), list_in_channels, list_out_channels):
            if i == 0:
                modules.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=4,
                                                       stride=2,
                                                       padding=2),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True)))

            elif i < depth - 2:
                modules.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=4,
                                                       stride=2,
                                                       padding=2),
                                             norm_layer(out_channels, affine=False),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True)))

            elif i == depth - 2:
                modules.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=4,
                                                       stride=1,
                                                       padding=2),
                                             norm_layer(out_channels, affine=False),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True)))

            else:
                if sigmoid:
                    modules.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=4,
                                                           stride=1,
                                                           padding=2),
                                                 nn.Sigmoid()))
                else:
                    modules.append(nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=4,
                                                           stride=1,
                                                           padding=2)))

        for module in modules:
            initialize_weights(module)

        if get_all_features:
            self._net = ListModule(*modules)
        else:
            self._net = nn.Sequential(*modules)

    def forward(self, x):
        if not self._get_all_features:
            return self._net.forward(x)
        else:
            outputs = [x]
            for module in self._net:
                outputs.append(module.forward(outputs[-1]))
            return outputs[1:]


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
    def __init__(self, in_channels, out_channels, instance_norm=True):
        super(FeatureEncoder, self).__init__()
        self._out_channels = out_channels

        if instance_norm:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        
        list_in_channels = [in_channels, 16, 32, 64, 128, 256, 128, 64, 32, 16]
        list_out_channels = list_in_channels[1:] + [out_channels]
        depth = len(list_out_channels)
        
        modules = []
        for i, in_ch, out_ch in zip(range(depth), list_in_channels, list_out_channels):
            if i == 0:
                modules += [nn.ReflectionPad2d(3),
                            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                            norm_layer(out_ch, affine=False), nn.ReLU(True)]
            elif i == depth-1:
                modules += [nn.ReflectionPad2d(3),
                            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0), nn.Tanh()]
            elif in_ch < out_ch:
                modules += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                            norm_layer(out_ch, affine=False), nn.ReLU(True)]
            elif in_ch > out_ch:
                modules += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(out_ch, affine=False), nn.ReLU(True)]

        self._net = nn.Sequential(*modules)
        initialize_weights(self._net)
        
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


class ResNetBlock(nn.Module):
    """
    ResNet Block for Generator.
    """

    def __init__(self, in_channels, norm_layer):
        super(ResNetBlock, self).__init__()

        self._resnetblock = nn.Sequential(nn.ReflectionPad2d(1),
                                          nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                                    stride=1, padding=0),
                                          norm_layer(in_channels, affine=False),
                                          nn.ReLU(inplace=True),
                                          nn.ReflectionPad2d(1),
                                          nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                                    stride=1, padding=0),
                                          norm_layer(in_channels, affine=False))
        initialize_weights(self._resnetblock)

    def forward(self, x):
        return x + self._resnetblock.forward(x)


class GlobalGenerator(nn.Module):
    """
    Global Generator from pix2pixHD
    """

    def __init__(self, in_channels, out_channels, instance_norm=True, num_residual_blocks=9):
        super(GlobalGenerator, self).__init__()
        if instance_norm:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        # convolutional front-end
        front_end = [nn.ReflectionPad2d(3),
                     nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=0),
                     norm_layer(64, affine=False),
                     nn.ReLU(inplace=True)]


        # downsampling layers
        downsampling = []

        in_channels_downsampling = [64, 128, 256, 512]
        out_channels_downsampling = [128, 256, 512, 1024]
        depth = len(in_channels_downsampling)

        for i, in_channel, out_channel in zip(range(depth), in_channels_downsampling,
                                              out_channels_downsampling):
            downsampling += [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
                             norm_layer(out_channel, affine=False),
                             nn.ReLU(inplace=True)]


        # resnet blocks
        resnetblocks = []
        resnet_channels = out_channels_downsampling[-1]

        for _ in range(num_residual_blocks):
            resnetblocks += [ResNetBlock(resnet_channels, norm_layer)]


        # upsampling layers
        in_channels_upsampling = [1024, 512, 256, 128]
        out_channels_upsampling = [512, 256, 128, 64]

        upsampling = []

        for i, in_channel, out_channel in zip(range(depth), in_channels_upsampling,
                                              out_channels_upsampling):
            upsampling += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2,
                                              padding=1, output_padding=1),
                           norm_layer(out_channel, affine=False),
                           nn.ReLU(inplace=True),
                           ]

        back_end = [nn.ReflectionPad2d(3),
                    nn.Conv2d(out_channels_upsampling[-1], out_channels, kernel_size=7, padding=0),
                    nn.Tanh()]

        self.model = nn.Sequential(*(front_end + downsampling + resnetblocks + upsampling))
        initialize_weights(self.model)
        self.last_block = nn.Sequential(*back_end)
        initialize_weights(self.last_block)

    def forward(self, x):
        output_model = self.model.forward(x)
        return self.last_block.forward(output_model)


class LocalEnhancer(nn.Module):
    """
    Local Enhancer Network in pix2pixHD
    """
    def __init__(self, in_channels, out_channels, num_resnet_blocks_enhancer=3,
                 num_resnet_blocks_global=9, path_to_global_generator_parameters=None,
                 instance_norm=True):
        super(LocalEnhancer, self).__init__()

        if instance_norm:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self._front_end = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(in_channels, 32, kernel_size=7, stride=1,
                                                  padding=0),
                                        norm_layer(32, affine=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                        norm_layer(64, affine=False),
                                        nn.ReLU(inplace=True))

        initialize_weights(self._front_end)

        global_generator = GlobalGenerator(in_channels, out_channels, instance_norm=instance_norm,
                                           num_residual_blocks=num_resnet_blocks_global)

        if path_to_global_generator_parameters:
            global_generator.load_state_dict(torch.load(path_to_global_generator_parameters))

        self._global_generator = global_generator.model

        residual_blocks = []

        for _ in range(num_resnet_blocks_enhancer):
            residual_blocks += [ResNetBlock(64, norm_layer)]

        self._residual_blocks = nn.Sequential(*residual_blocks)

        self._back_end = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                                          padding=1, output_padding=1),
                                       norm_layer(32, affine=False),
                                       nn.ReLU(inplace=True),
                                       nn.ReflectionPad2d(3),
                                       nn.Conv2d(32, out_channels, kernel_size=7, padding=0),
                                       nn.Tanh()
                                       )
        initialize_weights(self._back_end)

        self._downsampling = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        downsampled_input = self._downsampling(x)

        global_generator_output = self._global_generator.forward(downsampled_input)
        front_end_output = self._front_end.forward(x)

        residual_blocks_output = self._residual_blocks.forward(global_generator_output + front_end_output)

        return self._back_end.forward(residual_blocks_output)
