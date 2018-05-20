import torch
import torch.nn as nn


class DownModule(nn.Module):
    """
    Class for downsample module of the generator.
    """
    def __init__(self, in_channels, out_channels, batch_norm=True, stride=2,
                 relu=True, instance_norm=False):
        """
        Initialize down module and modules weights
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param batch_norm: whether to use batchnorm
        """
        super(DownModule, self).__init__()
        modules = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=4, stride=stride, padding=1, bias=False)]
        if batch_norm:
            modules.append(nn.BatchNorm2d(num_features=out_channels))
            
        if instance_norm:
            modules.append(nn.InstanceNorm2d(num_features=out_channels, affine=False))

        if relu:
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*modules)

        # weights initialization
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.fill_(0)

    def forward(self, x):
        return self.net.forward(x)




class UpModule(nn.Module):
    """
    Class for upsampling module in the generator.
    """
    def __init__(self, in_channels, out_channels, dropout=True, batch_norm=True,
                 relu=True, instance_norm=False, stride=2):
        super(UpModule, self).__init__()
        modules = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=4, stride=stride, padding=1, bias=False)]
        if batch_norm:
            modules.append(nn.BatchNorm2d(num_features=out_channels))
        if instance_norm:
            modules.append(nn.InstanceNorm2d(num_features=out_channels, affine=False))
        if dropout:
            modules.append(nn.Dropout2d(p=0.5, inplace=True))

        if relu:
            modules.append(nn.ReLU(inplace=True))
        else:
            modules.append(nn.Tanh())

        self.net = nn.Sequential(*modules)

        # weight initialization
        for module in self.net.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.fill_(0)

    def forward(self, x):
        return self.net.forward(x)


class ListModule(nn.Module):
    """
    Class for iterating modules
    """
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Generator(nn.Module):
    """
    pix2pix generator.
    """
    def __init__(self, in_channels, out_channels, instance_norm=False):
        super(Generator, self).__init__()

        list_in_channels_encoder = [in_channels, 64, 128, 256, 512, 512, 512, 512]
        list_out_channels_encoder = [64, 128, 256, 512, 512, 512, 512, 512]
        depth = len(list_out_channels_encoder)
        
        if instance_norm:
            batch_norm = False
        else:
            batch_norm = True

        down_modules = []

        for i, current_in_channels, current_out_channels in zip(range(depth),
                                                                list_in_channels_encoder,
                                                                list_out_channels_encoder):
            if i == 0 or i == depth - 1:
                down_modules.append(DownModule(current_in_channels, current_out_channels,
                                               batch_norm=False, instance_norm=False))
            else:
                down_modules.append(DownModule(current_in_channels, current_out_channels, 
                                               batch_norm=batch_norm, instance_norm=instance_norm))

        self._down_modules = ListModule(*down_modules)

        list_in_channels_decoder = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        list_out_channels_decoder = [512, 512, 512, 512, 256, 128, 64, out_channels]

        up_modules = []
        for i, current_in_channels, current_out_channels in zip(range(depth),
                                                                list_in_channels_decoder,
                                                                list_out_channels_decoder):
            if i < 3:
                up_modules.append(UpModule(current_in_channels, current_out_channels,
                                           batch_norm=batch_norm,
                                           instance_norm=instance_norm))
            elif i == depth - 1:
                up_modules.append(UpModule(current_in_channels, current_out_channels,
                                           dropout=False, batch_norm=False, relu=False,
                                           instance_norm=False))
            else:
                up_modules.append(UpModule(current_in_channels, current_out_channels,
                                           dropout=False, batch_norm=batch_norm,
                                           instance_norm=instance_norm))

        self._up_modules = ListModule(*up_modules)

    def forward(self, x):
        outputs = []
        current_input = x
        for module in self._down_modules:
            outputs.append(module.forward(current_input))
            current_input = outputs[-1]

        for i, module in enumerate(self._up_modules):
            idx_current_input = len(self._up_modules) - i - 1
            if i > 0:
                current_input = torch.cat((current_input, outputs[idx_current_input]), 1)

            current_input = module.forward(current_input)
        return current_input


class Discriminator(nn.Module):
    """
    Discriminator class for pix2pix model
    70x70 discriminator is used.
    """
    def __init__(self, in_channels, instance_norm=False, get_all_features=False):
        super(Discriminator, self).__init__()

        self._get_all_features = get_all_features

        list_in_channels = [in_channels, 64, 128, 256, 512]
        list_out_channels = [64, 128, 256, 512, 1]
        depth = len(list_out_channels)
        
        if instance_norm:
            batch_norm = False
        else:
            batch_norm = True

        modules = []

        for i, in_channels, out_channels in zip(range(depth), list_in_channels, list_out_channels):
            if i == 0:
                modules.append(DownModule(in_channels, out_channels, batch_norm=False,
                                          instance_norm=False))

            elif i < depth - 2:
                modules.append(DownModule(in_channels, out_channels, batch_norm=batch_norm,
                                          instance_norm=instance_norm))

            elif i == depth - 2:
                modules.append(DownModule(in_channels, out_channels, stride=1,
                                          batch_norm=batch_norm, instance_norm=instance_norm))

            else:
                modules.append(DownModule(in_channels, out_channels, stride=1, batch_norm=False,
                                          relu=False, instance_norm=False))

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
