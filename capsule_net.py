# -*- coding: utf-8 -*-
"""
@Time   : 2017/12/23 16:14
@Author : Nico
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from capsule_layers import CapsuleOut, CapsuleHidden, capsule_loss


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleHidden(num_inputs=256, num_outputs=32, vec_len=8)
        self.digit_capsules = CapsuleOut(num_inputs=8, num_outputs=10, vec_len=16, num_route_nodes=6 * 6 * 32, num_iterations=3)
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze()

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            if torch.cuda.is_available():
                y = Variable(torch.eye(10)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.eye(10)).index_select(dim=0, index=max_length_indices)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


if __name__ == '__main__':
    from keras.datasets import mnist
    from keras.utils import to_categorical
    from torch.optim import Adam
    from torch.utils import data
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = torch.from_numpy(x_train).float().unsqueeze(1)
    x /= 255.
    y = torch.from_numpy(to_categorical(y_train)).float()

    train_tensor = data.TensorDataset(data_tensor=x, target_tensor=y)
    train_loader = data.DataLoader(dataset=train_tensor, batch_size=128, shuffle=True, num_workers=0)

    model = CapsuleNet()
    model.cuda()
    optimizer = Adam(model.parameters())
    print model

    n_epochs = 10  # or whatever

    for epoch in range(n_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 1):

            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels, requires_grad=False).cuda()

            optimizer.zero_grad()

            # in case you wanted a semi-full example
            classes, reconstructions = model(inputs)
            loss = capsule_loss(inputs, labels, classes, reconstructions)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % train_loader.batch_size == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / train_loader.batch_size))
                running_loss = 0.0
