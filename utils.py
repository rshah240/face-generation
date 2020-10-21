import torch
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()
def scale(x, feature_range = (-1,1)):
    ''' Scale the images between 0 and 1'''
    min,max = feature_range
    x = x*(max - min) + min
    return x

def real_loss(D_out, smooth = True):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)

    if smooth:
        labels = labels * 0.9
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def conv(in_channels, out_channels, kernel_size, stride = 2, padding = 1,
         batch_norm = True):
    layers = []
    conv_layers = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                            kernel_size = kernel_size, stride = stride,padding = padding,
                            bias = False)
    layers.append(conv_layers)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride =2, padding = 1,
           batch_norm = True):
    layers = []
    conv_layers = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels,
                                     kernel_size = kernel_size, stride = stride, padding = padding,
                                     bias = False)
    layers.append(conv_layers)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        # defining convolution layers
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4,batch_norm=False) # first layer
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2,conv_dim*4,4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)

        # final, fully connected layer
        self.fc = nn.Linear(conv_dim*8*2*2,1)

    def forward(self,x):

        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))

        out = out.view(-1,self.conv_dim*8*2*2)

        # final output layer
        out = self.fc(out)

        return out


class Generator(nn.Module):
    def __init__(self, conv_dim, z_size):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.z_size = z_size

        self.fc = nn.Linear(in_features=z_size, out_features=conv_dim*8*2*2)
        self.t_conv1 = deconv(conv_dim*8, conv_dim*4,4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv3 = deconv(conv_dim*2,conv_dim,4)
        self.t_conv4 = deconv(conv_dim,3,4,batch_norm=False)

    def forward(self, x):
        out = F.relu(self.fc(x))
        out = out.view(-1, self.conv_dim*8,2,2)

        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = torch.tanh(self.t_conv4(out))

        return out

def weights_init_normal(m):
    """
    :param m: A module layer in a network
    """
    classname = m.__class__.__name__

    if classname.find('Conv')!= -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)


def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU')
        D.cuda()
        G.cuda()

    else:
        print('Training on CPU')

    return D, G









