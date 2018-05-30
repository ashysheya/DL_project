import scipy.misc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import nyu_dataset_depth
import pix2pix_model
import tqdm
import Deeper_Depth_Prediction.pytorch.model as model
import Deeper_Depth_Prediction.pytorch.weights as weights

num_classes = 41

segmentation_dataset_train = nyu_dataset_depth.SegmentationDataset(transforms=nyu_dataset_depth.SegmentationTransform(), use_depth=True)
data_loader_train = DataLoader(segmentation_dataset_train, batch_size=1, shuffle=True, num_workers=1)

segmentation_dataset_validation = nyu_dataset_depth.SegmentationDataset(path_to_datafolder='./datasets/nyu/val/', 
                                                                  transforms=
                                                                  nyu_dataset_depth.SegmentationTransform(False), use_depth=True)
data_loader_val = DataLoader(segmentation_dataset_validation, batch_size=1, shuffle=False, num_workers=1)

generator = pix2pix_model.Generator(num_classes + 1, 3, instance_norm=False).cuda(0)
discriminator = pix2pix_model.Discriminator(num_classes + 1 + 3, instance_norm=False).cuda(0)
depth_model = model.Model(1).cuda(0)
depth_model.load_state_dict(weights.load_weights(depth_model, 'NYU_ResNet-UpProj.npy', torch.cuda.FloatTensor))
for param in depth_model.parameters():
    param.requires_grad = False

def berhu(generated_depth, ground_truth_depth):
    y = generated_depth - ground_truth_depth
    c = torch.max(torch.abs(y)).detach()/5
    idx_geq, idx_leq = (y > c).nonzero().data, (y <= c).nonzero().data

    # print(idx_geq.dim(), idx_leq.dim())

    x = y.clone()
    if idx_geq.dim() != 0:
    	x[idx_geq[:,0], idx_geq[:,1], idx_geq[:,2], idx_geq[:,3]] \
                                = torch.abs(y[idx_geq[:,0], idx_geq[:,1], idx_geq[:,2], idx_geq[:,3]])
    if idx_leq.dim() != 0:
    	x[idx_leq[:,0], idx_leq[:,1], idx_leq[:,2], idx_leq[:,3]] \
                                = (y[idx_leq[:,0], idx_leq[:,1], idx_leq[:,2], idx_leq[:,3]]**2 + c**2)/2/c 
    return x.mean()


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
        for image, segmentation, depth, depth_reshaped in data_loader_train:
            image = Variable(image.cuda(0), requires_grad=False)
            segmentation = Variable(segmentation.cuda(0), requires_grad=False)
            depth = Variable(depth.cuda(0), requires_grad=False)
            depth_reshaped = Variable(depth_reshaped.cuda(0), requires_grad=False)
            
            input_generator = torch.cat((segmentation, depth), 1)
            fake_image = generator.forward(input_generator)
            
            # discriminator step
            d_optimizer.zero_grad()
            
            fake_input_discriminator = torch.cat((segmentation, depth, fake_image), 1)
            real_input_discriminator = torch.cat((segmentation, depth, image), 1)
            
            predictions_fake = discriminator.forward(fake_input_discriminator.detach())
            predictions_real = discriminator.forward(real_input_discriminator)
            
            target_tensor_true_label = Variable(torch.ones(predictions_fake.shape).cuda(0), requires_grad=False)
            
            loss_discriminator_real = GANLoss(predictions_real, target_tensor_true_label)
            
            target_tensor_fake_label = Variable(torch.zeros(predictions_fake.shape).cuda(0), requires_grad=False)
            loss_discriminator_fake = GANLoss(predictions_fake, target_tensor_fake_label)
            
            loss_discriminator = (loss_discriminator_fake + loss_discriminator_real)*0.5
            loss_discriminator.backward()
            d_optimizer.step()
            
            discriminator_loss_history.append(loss_discriminator.data[0])
            
            # generator step
            g_optimizer.zero_grad()
            
            fake_input_discriminator = torch.cat((segmentation, depth, fake_image), 1)         
            predictions_fake = discriminator.forward(fake_input_discriminator)
            
            loss_generator_gan = GANLoss(predictions_fake, target_tensor_true_label)
            
            loss_generator_L1 = L1Loss(fake_image, image)*lambda_param

            fake_depth = depth_model.forward(fake_image)
            loss_generator_depth = nn.MSELoss()(fake_depth, depth_reshaped)
            
            loss_generator = loss_generator_L1 + loss_generator_gan + loss_generator_depth
            loss_generator.backward()
            
            generator_loss_history.append(loss_generator.data[0])
            g_optimizer.step()
            
        num_val_image = 0
        for image, segmentation, depth, depth_reshaped in data_loader_val:
            segmentation = Variable(segmentation.cuda(0), requires_grad=False, volatile=True) 
            depth = Variable(depth.cuda(0), requires_grad=False, volatile=True)
            input_generator = torch.cat((segmentation, depth), 1)  
            generated_image = generator.forward(input_generator)
            
            scipy.misc.imsave('./depth_results_l2/{}_{}.png'.format(epoch, num_val_image), 
                              np.rollaxis(generated_image.cpu().data.numpy()[0], 0, 3))
            num_val_image += 1
            if num_val_image > 10:
                break
        
        if epoch % save_each_epoch == 0:
            torch.save(generator.state_dict(), './depth_models_l2/generator_{}'.format(epoch))
            torch.save(discriminator.state_dict(), './depth_models_l2/discriminator_{}'.format(epoch))
        
        if epoch > num_iter:
            lr_decay = learning_rate/num_iter_decay
            current_learning_rate -= lr_decay        
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = current_learning_rate
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = current_learning_rate

    return generator_loss_history, discriminator_loss_history

g_loss, d_loss = train(data_loader_train, data_loader_val, generator, discriminator, lambda_param=100.0)

np.save('depth_generator_loss_l2', g_loss)
np.save('depth_discriminator_loss_l2', d_loss)
