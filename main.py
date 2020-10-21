import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import utils
import torch.optim as optim
import pickle as pkl
import numpy as np
import hyperparameters as hp

train_on_gpu = torch.cuda.is_available()
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    '''
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param image_size:The Square size of the image data (x,y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    '''

    image_path = './' + data_dir
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(image_path, transform)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size,
                              shuffle = True)
    return train_loader

D, G = utils.build_network(d_conv_dim= hp.d_conv_dim, g_conv_dim = hp.g_conv_dim, z_size = hp.z_size)

def train(n_epochs,print_every=50):
    '''Training Adversarial Networks'''
    samples = []
    sample_size = 16
    losses = []
    fixed_z = np.random.uniform(-1,1, size = (sample_size, hp.z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    train_loader = get_dataloader(batch_size = 64,image_size=32)
    d_optimizer = optim.Adam(D.parameters(), lr = hp.lr,betas = (hp.beta_1, hp.beta_2))
    g_optimizer = optim.Adam(G.parameters(),lr = hp.lr,betas = (hp.beta_1, hp.beta_2))
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images,_) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = utils.scale(real_images)

            if train_on_gpu:
                real_images = real_images.cuda()

            # discriminator training
            d_optimizer.zero_grad()

            out_real = D(real_images)
            d_real_loss = utils.real_loss(out_real)

            z = np.random.uniform(-1,1,size = (batch_size,hp.z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            out_fake = D(fake_images)
            d_fake_loss = utils.fake_loss(out_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # generator training
            g_optimizer.zero_grad()
            z = np.random.uniform(-1,1,size = (batch_size,hp.z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()

            fake_images = G(z)
            out_fake_g = D(fake_images)
            g_loss = utils.real_loss(out_fake_g)
            g_loss.backward()
            g_optimizer.step()

            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))

        G.eval()
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()
    with open('train_samples.pkl','wb') as f:
        pkl.dump(samples,f)
    PATH = './Face_Generator.pth'
    torch.save(G.state_dict(), PATH)

    return losses

if __name__ == '__main__':
    losses = train(n_epochs= hp.n_epochs)
