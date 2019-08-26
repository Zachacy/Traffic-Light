from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils

import cv2
from PIL import Image
import os, argparse
import numpy as np
import xml.etree.ElementTree as ET

import torchvision.models as models
from utils import progress_bar, get_mean_and_std
#from models import alexnet

parser = argparse.ArgumentParser(description='PyTorch LightStatusDataset Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--threshold', default=0.7, type=float, help='threshold')
parser.add_argument('--epoch', default=5, type=int, help='epoch')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--input', default='4.jpg', help='Input Image')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

classes = ('Red', 'Green', 'Yellow', 'Right', 'Left', 'Straight')


# Dataset
class LightStatusDataset(Dataset):
    def __init__(self, classes, root, list_file, transform=None):
        self.classes = classes
        self.root = root
        self.transform = transform

        train_indexs = open(self.root + list_file).read().strip().split()
        self.train_lists = []
        for train_index in train_indexs:
            self.train_lists.append(train_index)

    def __len__(self):
        return len(self.train_lists)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, 'JPEGImages/', self.train_lists[idx])
        image = Image.open(img_name).convert('RGB')

        gt_name = os.path.join(self.root, 'Annotations/', self.train_lists[idx])
        gt_file = open(gt_name[:-4] + '.xml')
        tree = ET.parse(gt_file)
        root = tree.getroot()

        label = np.zeros(len(self.classes))
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in self.classes:
                continue
            label += np.eye(len(self.classes))[self.classes.index(cls)]
        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    transforms.ToTensor(),
    #transforms.Normalize((0.4457, 0.4833, 0.4989), (0.2537, 0.2547, 0.2529)) # new dataset
    #transforms.Normalize((0.4457, 0.4833, 0.4989), (0.2537, 0.2547, 0.2529)) # new dataset + ITRI new dataset
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4457, 0.4833, 0.4989), (0.2537, 0.2547, 0.2529)) # new dataset
    #transforms.Normalize((0.4457, 0.4833, 0.4989), (0.2537, 0.2547, 0.2529)) # new dataset + ITRI new dataset
])


trainset = LightStatusDataset(classes=classes, root='/home/rvl/Dataset/vatic/TL_classify/', list_file='train.txt', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

testset = LightStatusDataset(classes=classes, root='/home/rvl/Dataset/New_TL/', list_file='test.txt', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4)


# Model
print('==> Building model..')
#net = alexnet.AlexNet(num_classes=len(classes))

net = models.alexnet(pretrained=False)
net.classifier  = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, len(classes)),
)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs * 255
        optimizer.zero_grad()
        outputs = net(inputs) # input.size() = (b, c, h, w)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (F.sigmoid(outputs) >= args.threshold).float()
        total += targets.size(0) * targets.size(1)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs * 255
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = (F.sigmoid(outputs) >= args.threshold).float()
            total += targets.size(0) * targets.size(1)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def test_single_image(input_path=None):
    net.eval()
    with torch.no_grad():
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

        #image = cv2.imread(input_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image, 'RGB')
        image = Image.open(input_path).convert('RGB')

        image = transform_test(image)
        image = torch.unsqueeze(image, 0)
        inputs = image.to(device)
        inputs = inputs * 255

        outputs = net(inputs)
        predicted = F.sigmoid(outputs)

        print('Inpu Image : %s' %(input_path))
        print('classes', classes)
        print('Predicted :', " ".join(('%.3f') % x for x in predicted[0]))


def main():
    '''
    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(epoch)
        test(epoch)
    '''
    #test_single_image(input_path='115.jpg')
    test_single_image(input_path='test.png')
    test_single_image(input_path='test_2.png')
    test_single_image(input_path=args.input)

    #mean, std = get_mean_and_std(trainset)
    #print(mean, std)

if __name__ == '__main__':
    main()


