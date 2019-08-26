from __future__ import print_function

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torchvision
import os,cv2
import argparse
from models import alexnet_NIN
from torchsummary import summary
from utils import progress_bar
import torchvision.models as models
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--threshold', default=0.7, type=float, help='learning rate')
parser.add_argument('--input', default='3_1.jpg', help='Input Image')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parser.parse_args()

classes = ['Red', 'Yellow', 'Green', 'Left','Right', 'Straight']
img_size = [224,224]
print('==> Building model..')
net = alexnet_NIN.AlexNet(num_classes=2)
net = net.to(device)
summary(net,(3,224,224))
'''
net.eval()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


#Load checkpoint
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt_NIN.t7')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

if __name__=='__main__':

    img = Image.open(args.input).resize((img_size[0],img_size[1]), Image.ANTIALIAS)
    trans = transforms.ToTensor()
    inputs = trans(img).unsqueeze_(0).to(device)
    #print(inputs.shape, inputs[0,0,0,0])
    outputs = net(inputs*255.0)
    predicted = F.sigmoid(outputs)
    print('Inpu Image : %s' %(args.input))
    print('classes',classes)
    print('Predicted :',predicted)
    img.show()
'''