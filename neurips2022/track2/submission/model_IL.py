import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):

        return x.view(x.size()[0], -1)


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 4, 4)
        self.bachnorm1 = torch.nn.BatchNorm2d(32)
        self.elu = torch.nn.ELU()
        self.dropout2d = torch.nn.Dropout2d(0.5)
        self.conv2 = torch.nn.Conv2d(32, 64, 2, 2)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 4, 4)
        self.flatten = Flatten()
        self.batchnorm3 = torch.nn.BatchNorm1d(64 * 8 * 8)
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.batchnorm4 = torch.nn.BatchNorm1d(128)
        self.linear2 = torch.nn.Linear(128, 32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bachnorm1(x)
        x = self.elu(x)
        x = self.dropout2d(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.dropout2d(x)

        x = self.conv3(x)
        x = self.elu(x)
        x = self.flatten(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.elu(x)
        x = self.batchnorm4(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.elu(x)

        return x


class EmbBlock(nn.Module):
    def __init__(self):
        super(EmbBlock, self).__init__()

        self.linear1 = torch.nn.Linear(2, 32)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.elu(x)

        return x


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.convblock = ConvBlock()
        self.embblock = EmbBlock()

        self.linear1 = torch.nn.Linear(64, 32)
        self.linear4 = torch.nn.Linear(32, 1)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear5 = torch.nn.Linear(32, 1)
        self.linear3 = torch.nn.Linear(64, 32)
        self.linear6 = torch.nn.Linear(32, 1)

        self.elu = torch.nn.ELU()

        self.criterion = nn.MSELoss()

    def forward(self, x):
        img_emb = self.convblock(x[0])
        goal_emb = self.embblock(x[1])

        x = torch.cat((img_emb, goal_emb), dim=1)

        dx = self.linear1(x)
        dy = self.linear2(x)
        dheading = self.linear3(x)

        dx = self.elu(dx)
        dy = self.elu(dy)
        dheading = self.elu(dheading)

        nn_outputs = {}
        nn_outputs["dx"] = self.linear4(dx)
        nn_outputs["dy"] = self.linear5(dy)
        nn_outputs["d_heading"] = self.linear6(dheading)

        return nn_outputs

    def compute_loss(self, nn_outputs, labels):
        dx_loss = self.criterion(
            nn_outputs["dx"], labels[:, 0].view(labels.shape[0], 1)
        )
        dy_loss = self.criterion(
            nn_outputs["dy"], labels[:, 1].view(labels.shape[0], 1)
        )
        dh_loss = self.criterion(
            nn_outputs["d_heading"], labels[:, 2].view(labels.shape[0], 1)
        )
        total_loss = dx_loss + dy_loss + dh_loss

        return total_loss, dx_loss, dy_loss, dh_loss
