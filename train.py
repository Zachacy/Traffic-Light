'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os,cv2
import argparse
import load_data
from models import alexnet
from models import mobilenet
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch classification Training')
parser.add_argument('--threshold', default=0.7, type=float, help='learning rate')
parser.add_argument('--train_index', default='trainval.txt', type=str, help='train index list')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=10, type=int, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=80, type=int, help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.5  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
image_size = [200,75]
Image_path = '/home/rvl/Dataset/traffic_light_data/Traffic_light_classify/JPEGImages'
Annota_path = '/home/rvl/Dataset/traffic_light_data/Traffic_light_classify/instances.json'
classes = ['Red', 'Yellow', 'Green', 'Left','Right', 'Straight']
# Data
print('==> Preparing data..')
transform = transforms.Compose([ \
    transforms.Resize((image_size[0],image_size[1])), \
    transforms.ToTensor()])

Data = datasets.CocoDetection(root=Image_path, \
    annFile=Annota_path, \
    transform=transform)

trainloader = torch.utils.data.DataLoader(Data, batch_size=1, shuffle=True, num_workers=0)


# Model
print('==> Building model..')
net = alexnet.AlexNet(num_classes=len(classes))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.MultiLabelSoftMarginLoss()
#criterion = nn.MultiLabelMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

# Training
def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    imgs = torch.zeros([1,3,image_size[0],image_size[1]])
    gt_labels = torch.zeros([1,len(classes)])

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs * 255.0
        cls_id = np.zeros([1,len(classes)])
        for i in range(0,len(targets)):    
            cls_id += np.eye(len(classes))[int(targets[i]['category_id'])-1]
        if torch.sum(gt_labels) == 0 :
            imgs = inputs
            gt_labels = torch.Tensor(cls_id)
        else :
            imgs = torch.cat([imgs, inputs], dim = 0)
            gt_labels = torch.cat([gt_labels, torch.Tensor(cls_id)], dim = 0)
        if imgs.shape[0] == args.batch_size:
            imgs, gt_labels = imgs.to(device), gt_labels.to(device)#target = [batch * class_num], input = [batch * image]
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, gt_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() # type(train_loss) = float
            predicted = (F.sigmoid(outputs) >= args.threshold).float()
            total += gt_labels.size(0) * gt_labels.size(1)
            correct += predicted.eq(gt_labels).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            imgs = torch.zeros([1,3,image_size[0],image_size[1]])
            gt_labels = torch.zeros([1,len(classes)])
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_NIN.t7')
        best_acc = acc

for epoch in range(0, args.epoch):
    train(epoch)
