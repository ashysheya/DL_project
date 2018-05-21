import scipy.misc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import nyu_dataset
import pix2pix_model
import tqdm

num_classes = 41

segmentation_dataset_train = nyu_dataset.SegmentationDataset(transforms=nyu_dataset.SegmentationTransform())
data_loader_train = DataLoader(segmentation_dataset_train, batch_size=1, shuffle=True, num_workers=1)

segmentation_dataset_validation = nyu_dataset.SegmentationDataset(path_to_datafolder='./datasets/nyu/val/', 
                                                                  transforms=
                                                                  nyu_dataset.SegmentationTransform(False))
data_loader_val = DataLoader(segmentation_dataset_validation, batch_size=1, shuffle=False, num_workers=1)

generator = pix2pix_model.Generator(num_classes, 3, instance_norm=False).cuda(0)
discriminator = pix2pix_model.Discriminator(num_classes + 3, instance_norm=False).cuda(0)

def train(data_loader_train, data_loader_val, generator, discriminator, num_iter=100, num_iter_decay=100, 
          lambda_param=100.0, save_each_epoch=10, learning_rate=0.0002):
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    current_learning_rate = learning_rate
    
    GANLoss = nn.BCELoss()
    L1Loss = nn.L1Loss()
    
    generator_loss_history = []
    discriminator_loss_history = []
    
    for epoch in tqdm.tqdm(range(1, num_iter + num_iter_decay + 1)):
        for image_batch, segmentation_batch in data_loader_train:
            image_batch = Variable(image_batch.cuda(0), requires_grad=False)
            segmentation_batch = Variable(segmentation_batch.cuda(0), requires_grad=False)
            
            fake_image_batch = generator.forward(segmentation_batch)
            
            # discriminator step
            d_optimizer.zero_grad()
            
            fake_input_discriminator = torch.cat((segmentation_batch, fake_image_batch), 1)
            real_input_discriminator = torch.cat((segmentation_batch, image_batch), 1)
            
            predictions_fake = discriminator.forward(fake_input_discriminator.detach())
            predictions_real = discriminator.forward(real_input_discriminator)
            
            target_tensor = Variable(torch.ones(predictions_fake.shape).cuda(0), requires_grad=False)
            
            loss_discriminator_real = GANLoss(predictions_real, target_tensor)
            
            target_tensor.data.fill_(0)
            loss_discriminator_fake = GANLoss(predictions_fake, target_tensor)
            
            loss_discriminator = (loss_discriminator_fake + loss_discriminator_real)*0.5
            loss_discriminator.backward()
            d_optimizer.step()
            
            discriminator_loss_history.append(loss_discriminator.data[0])
            
            # generator step
            g_optimizer.zero_grad()
            
            fake_input_discriminator = torch.cat((segmentation_batch, fake_image_batch), 1)         
            predictions_fake = discriminator.forward(fake_input_discriminator)
            
            target_tensor.data.fill_(1)
            loss_generator_gan = GANLoss(predictions_fake, target_tensor)
            
            loss_generator_L1 = L1Loss(fake_image_batch, image_batch)*lambda_param
            
            loss_generator = loss_generator_L1 + loss_generator_gan
            loss_generator.backward()
            
            generator_loss_history.append(loss_generator.data[0])
            g_optimizer.step()
            
        num_val_image = 0
        for image, segmentation in data_loader_val:
            segmentation = Variable(segmentation.cuda(0), requires_grad=False, volatile=True)   
            generated_image = generator.forward(segmentation)
            
            scipy.misc.imsave('./results/{}_{}.png'.format(epoch, num_val_image), 
                              np.rollaxis(generated_image.cpu().data.numpy()[0], 0, 3))
            num_val_image += 1
            if num_val_image > 10:
                break
        
        if epoch % save_each_epoch == 0:
            torch.save(generator.state_dict(), './models/generator_{}'.format(epoch))
            torch.save(discriminator.state_dict(), './models/discriminator_{}'.format(epoch))
        
        if epoch > num_iter:
            lr_decay = learning_rate/num_iter_decay
            current_learning_rate -= lr_decay        
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = current_learning_rate
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = current_learning_rate

    return generator_loss_history, discriminator_loss_history

g_loss, d_loss = train(data_loader_train, data_loader_val, generator, discriminator, lambda_param=100.0)

np.save('generator_loss', g_loss)
np.save('discriminator_loss', d_loss)
