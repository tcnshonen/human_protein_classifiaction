import torch
from torch import nn
import torch.nn.functional as F


def activation(name):
    if name == 'elu':
        act = nn.elu()
    elif name == 'leakyrelu':
        act = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        act = nn.Tanh()
    elif name == 'sigmoid':
        act = nn.Sigmoid()
    else:
        act = nn.ReLU()
    return act


class LinearLayer(nn.Module):
    def __init__(self, in_num, out_num, norm=True, dropout=True, activation_name='relu'):
        super(LinearLayer, self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.activation_name = activation_name
        
        modules = [nn.Linear(in_num, out_num)]
        
        if self.norm:
            modules.append(nn.BatchNorm1d(out_num))
            
        if self.activation_name:
            modules.append(activation(activation_name))
        
        if self.dropout:
            modules.append(nn.Dropout(p=0.2))
            
        self.layers = nn.Sequential(*modules)
        
            
    def forward(self, x):
        x = self.layers(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_num, out_num, kernel=1, stride=1, padding=0, output_padding=0, norm=True, dropout=True, activation_name='relu', transpose=False):
        super(ConvLayer, self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.activation_name = activation_name
        self.transpose = transpose
        self.config = {'kernel_size': kernel, 'stride': stride, 'padding': padding}
        
        modules = []
        if self.transpose:
            self.config['output_padding'] = output_padding
            modules.append(nn.ConvTranspose2d(in_num, out_num, **self.config))
        else:
            modules.append(nn.Conv2d(in_num, out_num, **self.config))
            
        if self.norm:
            modules.append(nn.BatchNorm2d(out_num))
        
        modules.append(activation(activation_name))
            
        if self.dropout:
            modules.append(nn.Dropout2d(p=0.2))
            
        self.layers = nn.Sequential(*modules)
            
            
    def forward(self, x):
        x = self.layers(x)
        return x
    

class Net(nn.Module):
    def __init__(self, norm=True, dropout=True):
        super(Net, self).__init__()
        
        self.norm = norm
        self.dropout = dropout
        self.config = {'norm': self.norm, 'dropout': self.dropout}
        
        self.convs1 = nn.Sequential(
            ConvLayer(3, 8, 3, stride=1, padding=1, **self.config),
            ConvLayer(8, 8, 3, stride=1, padding=1, **self.config),
            ConvLayer(8, 16, 3, stride=2, padding=1, **self.config),
            ConvLayer(16, 16, 3, stride=1, padding=1, **self.config),
            ConvLayer(16, 16, 3, stride=1, padding=1, **self.config),
            ConvLayer(16, 32, 3, stride=2, padding=1, **self.config),
            ConvLayer(32, 32, 3, stride=1, padding=1, **self.config),
            ConvLayer(32, 32, 3, stride=1, padding=1, **self.config),
        )
        
        self.convs2 = nn.Sequential(
            ConvLayer(32, 64, 3, stride=2, padding=1, **self.config),
            ConvLayer(64, 64, 3, stride=1, padding=1, **self.config),
            ConvLayer(64, 64, 3, stride=1, padding=1, **self.config),
        )
        
        self.convs3 = nn.Sequential(
            ConvLayer(64, 128, 3, stride=2, padding=1, **self.config),
            ConvLayer(128, 128, 3, stride=1, padding=1, **self.config),
            ConvLayer(128, 128, 3, stride=1, padding=1, **self.config),
        )
        
        self.conv_appearance = ConvLayer(128, 128, 3, stride=2, padding=1, **self.config)
        self.conv4 = ConvLayer(128, 64, 3, stride=2, padding=1, output_padding=1, transpose=True, **self.config)
        
        self.convs5 = nn.Sequential(
            ConvLayer(128, 64, 3, stride=1, padding=1, **self.config),
            ConvLayer(64, 64, 3, stride=1, padding=1, **self.config),
            ConvLayer(64, 32, 3, stride=2, padding=1, output_padding=1, transpose=True, **self.config),
        )
        
        self.convs6 = nn.Sequential(
            ConvLayer(64, 32, 3, stride=1, padding=1, **self.config),
            ConvLayer(32, 32, 3, stride=1, padding=1, **self.config),
        )
        
        conv_list = []
        for _ in range(28):
            conv_list.append(ConvLayer(32, 1, 3, stride=1, padding=1, activation_name='sigmoid', **self.config))
        self.conv_list = nn.ModuleList(conv_list)
        
        self.convs7 = nn.Sequential(
            ConvLayer(28, 56, 3, stride=2, padding=1, **self.config),
            ConvLayer(56, 112, 3, stride=2, padding=1, **self.config),
            ConvLayer(112, 224, 3, stride=2, padding=1, **self.config),
        )
        
        self.fc = LinearLayer(352, 28, activation_name=None, **self.config)
        
        
    def forward(self, x):
        x_32d = self.convs1(x)
        x_64d = self.convs2(x_32d)
        x_128d = self.convs3(x_64d)
        x_appearance = self.conv_appearance(x_128d)
        x_64d_2 = self.conv4(x_128d)
        x_128d_2 = torch.cat((x_64d, x_64d_2), 1)
        x_32d_2 = self.convs5(x_128d_2)
        
        x_64d_3 = torch.cat((x_32d, x_32d_2), 1)
        x_64d_3 = self.convs6(x_64d_3)
        
        x_feature = []
        for conv in self.conv_list:
            x_feature.append(conv(x_64d_3))
        x_feature = torch.cat(x_feature, 1)
        
        x_feature = self.convs7(x_feature)
        x_feature = torch.cat((x_appearance, x_feature), 1)
        
        x_feature = nn.AvgPool2d(16)(x_feature)
        x_feature = x_feature.view(-1, 352)
        x_feature = self.fc(x_feature)
        
        return x_feature
