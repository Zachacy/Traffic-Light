from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import torchvision
import os,cv2
import argparse
import load_data
from models import alexnet
from models import mobilenet
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--threshold', default=0.7, type=float, help='learning rate')
parser.add_argument('--input', default='test_image/3_1.jpg', help='Input Image')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parser.parse_args()

classes = ['Red', 'Yellow', 'Green', 'Left','Right', 'Straight']
img_size = 224
print('==> Building model..')
net = alexnet.AlexNet(num_classes=len(classes))
#net = mobilenet.MobileNet(num_classes=4)
net = net.to(device)
net.eval()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


#Load checkpoint
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt_old_i.t7')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

if __name__=='__main__':
    img = cv2.imread(args.input)
    img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)
    img = np.expand_dims(img, axis=0)
    inputs = (torch.tensor(img)).permute(0, 3, 1, 2).float().to(device)
    outputs = net(inputs)
    predicted = F.sigmoid(outputs)
    print('Inpu Image : %s' %(args.input))
    print('classes',classes)
    print('Predicted : ', predicted)
