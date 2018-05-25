import scipy.misc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import nyu_dataset
import pix2pixHD_model
import tqdm
import matplotlib.pyplot as plt
# % matplotlib inline

def train(data_loader_train, data_loader_val, generator, discriminator, encoder, num_iter=100, num_iter_decay=100, 
    lambda_param=10.0, save_each_epoch=10, learning_rate=0.0002, _cuda=0):
        e_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        current_learning_rate = learning_rate

        GANLoss = nn.BCELoss()

        def FMLoss(disc_real_list, disc_fake_list):
            loss = 0
            for disc_real, disc_fake in zip(disc_real_list, disc_fake_list):
                for layer_real, layer_fake in zip(disc_real, disc_fake):
                    loss += nn.L1Loss()(layer_fake, layer_real.detach())
            return loss

        generator_loss_history = []
        discriminator_loss_history = []

        for epoch in tqdm.tqdm(range(1, num_iter + num_iter_decay + 1)):
            for image_batch, segmentation_batch, inst_batch, border_batch in data_loader_train:
                image_batch = Variable(image_batch.cuda(_cuda), requires_grad=False)
                segmentation_batch = Variable(segmentation_batch.cuda(_cuda), requires_grad=False)
                border_batch = Variable(border_batch.cuda(_cuda), requires_grad=False)

                encoded = encoder.forward(image_batch, inst_batch)
                input_generator = torch.cat((segmentation_batch, encoded, border_batch), 1)
                fake_image_batch = generator.forward(input_generator)

                # discriminator step
                d_optimizer.zero_grad()

                fake_input_discriminator = torch.cat((segmentation_batch, border_batch, fake_image_batch), 1)
                real_input_discriminator = torch.cat((segmentation_batch, border_batch, image_batch), 1)

                predictions_fake = discriminator.forward(fake_input_discriminator.detach())
                predictions_real = discriminator.forward(real_input_discriminator)

                loss_discriminator = 0
                for disc_real, disc_fake in zip(predictions_real, predictions_fake):
                    target_tensor_real_label = Variable(torch.ones(disc_real[-1].shape).cuda(_cuda), requires_grad=False)
                    loss_discriminator_real = GANLoss(disc_real[-1], target_tensor_real_label)

                    target_tensor_fake_label = Variable(torch.zeros(disc_real[-1].shape).cuda(_cuda), requires_grad=False)
                    loss_discriminator_fake = GANLoss(disc_fake[-1], target_tensor_fake_label)

                    loss_discriminator += 0.5 * (loss_discriminator_fake + loss_discriminator_real)

                loss_discriminator.backward()
                d_optimizer.step()

                discriminator_loss_history.append(loss_discriminator.data[0])

                # generator & encoder step
                e_optimizer.zero_grad()
                g_optimizer.zero_grad()

                fake_input_discriminator = torch.cat((segmentation_batch, border_batch, fake_image_batch), 1)
                predictions_fake = discriminator.forward(fake_input_discriminator)

                loss_generator = 0
                # get GANLoss for generator
                for disc in predictions_fake:
                    target_tensor = Variable(torch.ones(disc[-1].shape).cuda(_cuda),requires_grad=False)
                    loss_generator += GANLoss(disc[-1], target_tensor)

                # get FMLoss for generator
                predictions_real = discriminator.forward(real_input_discriminator)
                loss_generator += lambda_param * FMLoss(predictions_fake, predictions_real)

                loss_generator.backward()

                generator_loss_history.append(loss_generator.data[0])
                g_optimizer.step()
                e_optimizer.step()

            num_val_image = 0
            for image, segmentation, inst, borders  in data_loader_val:                
                segmentation = Variable(segmentation.cuda(_cuda), requires_grad=False, volatile=True)
                borders = Variable(borders.cuda(_cuda), requires_grad=False, volatile=True)
                encoded_val = encoder.forward(image, inst)
                
                input_generator = torch.cat((segmentation, encoded_val, borders), 1)
                generated_image = generator.forward(input_generator)

                scipy.misc.imsave('./pix2pixHD/results/{}_{}.png'.format(epoch, num_val_image),
                                  np.rollaxis(generated_image.cpu().data.numpy()[0], 0, 3))
                num_val_image += 1
                if num_val_image > 10:
                    break

            if epoch % save_each_epoch == 0:
                torch.save(generator.state_dict(), './pix2pixHD/models/generator_{}_{}'.format(epoch, 0))
                torch.save(discriminator.state_dict(), './pix2pixHD/models/discriminator_{}_{}'.format(epoch, 0))
                torch.save(encoder.state_dict(), './pix2pixHD/models/encoder_{}_{}'.format(epoch, 0))

                np.save('./pix2pixHD/generator_loss', loss_generator)
                np.save('./pix2pixHD/discriminator_loss', loss_discriminator)

            if epoch > num_iter:
                lr_decay = learning_rate / num_iter_decay
                current_learning_rate -= lr_decay
                for param_group in d_optimizer.param_groups:
                    param_group['lr'] = current_learning_rate
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = current_learning_rate
                for param_group in e_optimizer.param_groups:
                    param_group['lr'] = current_learning_rate

        return generator_loss_history, discriminator_loss_history

if __name__ == "__main__": 
    num_classes = 3

    segmentation_dataset_train = nyu_dataset.SegmentationDataset(
        transforms=nyu_dataset.SegmentationTransform())
    data_loader_train = DataLoader(segmentation_dataset_train, batch_size=1, shuffle=True,
                                   num_workers=1)

    segmentation_dataset_validation = nyu_dataset.SegmentationDataset(
        path_to_datafolder='./datasets/nyu/val/',
        transforms=
        nyu_dataset.SegmentationTransform(False))
    data_loader_val = DataLoader(segmentation_dataset_validation, batch_size=1, shuffle=False,
                                 num_workers=1)

    _cuda = 2

    encoder = pix2pixHD_model.FeatureEncoder(num_classes, 3, instance_norm=True).cuda(_cuda)
    generator = pix2pixHD_model.GlobalGenerator(num_classes + 3 + 1, 3, instance_norm=True).cuda(_cuda)
    discriminator = pix2pixHD_model.MultiScaleDiscriminator(num_classes + 3, instance_norm=True).cuda(_cuda)
    

    g_loss, d_loss = train(data_loader_train, data_loader_val, generator, discriminator, encoder)