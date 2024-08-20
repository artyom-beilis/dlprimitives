###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import h5py
import time
import json


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.mjs = dict(
            inputs = [dict(shape=[128,1,28,28],name='data')],
            operators = [
                dict(name="fc1",
                     type="InnerProduct",
                     inputs=['data'],
                     outputs=['fc1'],
                     options = dict(
                        outputs=512,
                        activation='relu'
                     )
                ),
                dict(name="fc2",
                     type="InnerProduct",
                     inputs=['fc1'],
                     outputs=['fc2'],
                     options = dict(
                        outputs=256,
                        activation='relu'
                     )
                ),
                dict(name="fc3",
                     type="InnerProduct",
                     inputs=['fc2'],
                     outputs=['fc3'],
                     options = dict(
                        outputs=10,
                     )
                ),
                dict(name="prob",
                     type="SoftMax",
                     inputs=["fc3"],
                     outputs=["prob"]
                )
            ]
        )

    def save_dp_net(self,file_name,batch):
        self.mjs['inputs'][0]['shape'][0] = batch
        with open(file_name,'w') as f:
            json.dump(self.mjs,f,indent=4)

    def _add_fc(self,h,fc,name):
        for n,param in enumerate(fc.parameters()):
            ds = h.create_dataset('%s.%d' % (name,n),param.shape)
            ds[:] = param.cpu().data.numpy()
        
    def save_dp_weights(self,file_name):
        h = h5py.File(file_name,'w')
        self._add_fc(h,self.fc1,'fc1')
        self._add_fc(h,self.fc2,'fc2')
        self._add_fc(h,self.fc3,'fc3')
        h.close()

    def forward(self, x):
        x = F.relu(self.fc1(torch.flatten(x,1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.training:
            output = F.log_softmax(x, dim=1)
        else:
            output = F.softmax(x,dim=1)
        return output

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.cnv1 = nn.Conv2d(1, 32,5,padding=2)
        self.cnv2 = nn.Conv2d(32,64,5,padding=2)
        self.fc1 = nn.Linear(64 * 28 // 4 * 28 // 4, 256)
        #self.fc1 = nn.Linear(32 * 4*4, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.p1   = nn.MaxPool2d(2,stride=2)
        self.p2   = nn.MaxPool2d(2,stride=2)

        self.mjs = dict(
            inputs = [dict(shape=[64,1,28,28],name='data')],
            operators = [
                dict(name="cnv1",
                     type="Convolution2D",
                     inputs=["data"],
                     outputs=["cnv1"],
                     options = dict(
                        channels_out = 32,
                        kernel=list(self.cnv1.kernel_size),
                        pad=list(self.cnv1.padding),
                        activation="relu"
                    )
                ),
                dict(name="p1",
                     type="Pooling2D",
                     inputs=["cnv1"],outputs=["p1"],
                     options = dict(kernel=2,stride=2)),
                dict(name="cnv2",
                     type="Convolution2D",
                     inputs=["p1"],
                     outputs=["cnv2"],
                     options = dict(
                        channels_out = 64,
                        kernel=list(self.cnv2.kernel_size),
                        pad=list(self.cnv2.padding),
                        activation="relu"
                    )
                ),
                dict(name="p2",
                     type="Pooling2D",
                     inputs=["cnv2"],outputs=["p2"],
                     options = dict(kernel=2,stride=2)),
                dict(name="fc1",
                     type="InnerProduct",
                     inputs=['p2'],
                     outputs=['fc1'],
                     options = dict(
                        outputs=256,
                        activation='relu'
                     )
                ),
                dict(name="fc2",
                     type="InnerProduct",
                     inputs=['fc1'],
                     outputs=['fc2'],
                     options = dict(
                        outputs=10,
                     )
                ),
                dict(name="prob",
                     type="SoftMax",
                     inputs=["fc2"],
                     outputs=["prob"]
                )
            ]
        )

    def save_dp_net(self,file_name,batch):
        self.mjs['inputs'][0]['shape'][0] = batch
        with open(file_name,'w') as f:
            json.dump(self.mjs,f,indent=4)

    def _add_fc(self,h,fc,name):
        for n,param in enumerate(fc.parameters()):
            ds = h.create_dataset('%s.%d' % (name,n),param.shape)
            ds[:] = param.cpu().data.numpy()
        
    def save_dp_weights(self,file_name):
        h = h5py.File(file_name,'w')
        self._add_fc(h,self.cnv1,'cnv1')
        self._add_fc(h,self.cnv2,'cnv2')
        self._add_fc(h,self.fc1,'fc1')
        self._add_fc(h,self.fc2,'fc2')
        h.close()

    def forward(self, x):
        x = self.p1(F.relu(self.cnv1(x)))
        x = self.p2(F.relu(self.cnv2(x)))
        x = F.relu(self.fc1(torch.flatten(x,1)))
        x = self.fc2(x)
        if self.training:
            output = F.log_softmax(x, dim=1)
        else:
            output = F.softmax(x,dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    end = time.time()
    print("Training epch time ",end - start)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        total_time = 0
        total_items = 0
        for data, target in test_loader:
            start = time.time()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            end = time.time()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_time += end - start
            total_items += data.shape[0]

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(" %5.3fus per sample " % (total_time / total_items * 1e6))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device",device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize((0.0,), (1.0,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=0.00)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()

    if args.save_model:
        model.eval()
        torch.save(model.state_dict(), "mnist_cnn.pt")
        inp = torch.randn(args.test_batch_size,1,28,28).to(device)
        torch.onnx.export(model,inp,'mnist.onnx',verbose=True,input_names=['data'],output_names=['prob'])
        model.save_dp_net('mnist_dp.json',args.test_batch_size)
        model.save_dp_weights('mnist_dp.h5')


if __name__ == '__main__':
    main()
