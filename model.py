import torch
from torch import nn


class Pixel2Point(nn.Module):
    def __init__(self, initial_point=None):
        super(Pixel2Point, self).__init__()
        self.layer1 = self.conv_module(1, 32)
        self.layer2 = self.conv_module(32, 64)
        self.layer3 = self.conv_module(64, 128)
        self.layer4 = self.conv_module(128, 256)
        self.layer5 = self.conv_module(256, 256)
        self.layer6 = self.conv_module(256, 256)
        self.layer7 = self.conv_module(256, 256)
        self.fc1 = self.fc_module(256 * (3 + 256), 2048 * 5)
        self.fc2 = self.fc_module(2048 * 5, 2048 * 5)
        self.fc3 = self.fc_module(2048 * 5, 2048 * 4)
        self.fc4 = nn.Linear(2048 * 4, 2048 * 3)
        if initial_point == 'two_circle':
            self.initial_point = self.two_ball()
        else:
            self.initial_point = self.generate_initial_point()

    def forward(self, x):
        encoder_out = self.layer1(x)
        encoder_out = self.layer2(encoder_out)
        encoder_out = self.layer3(encoder_out)
        encoder_out = self.layer4(encoder_out)
        encoder_out = self.layer5(encoder_out)
        encoder_out = self.layer6(encoder_out)
        fv = self.layer7(encoder_out)
        batch_size = fv.shape[0]

        initial_pc_fv = torch.cat((
            self.initial_point.to(fv.device).view(1, 256, 3).repeat(batch_size, 1, 1),  # torch.Size([batch, 256, 3])
            fv.view(batch_size, 1, 256).repeat(1, 256, 1)  # torch.Size([batch, 256, 256])
        ), dim=2, )  # [batch_size, 256, 259]

        generator_out = self.fc1(initial_pc_fv.view(batch_size, -1))
        generator_out = self.fc2(generator_out)
        generator_out = self.fc3(generator_out)
        generator_out = self.fc4(generator_out)

        return generator_out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def fc_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Linear(in_num, out_num),
            nn.ReLU(),
        )

    def generate_initial_point(self):
        u = torch.linspace(0, 2 * torch.pi, 16)
        v = torch.linspace(0, torch.pi, 16)
        return torch.vstack((
            torch.flatten(1 * torch.outer(torch.cos(u), torch.sin(v))),
            torch.flatten(1 * torch.outer(torch.sin(u), torch.sin(v))),
            torch.flatten(1 * torch.outer(torch.ones(torch.Tensor.size(u)), torch.cos(v))))
        ).T

    def two_ball(self):
        one_ball = self.generate_initial_point()
        one_ball[1:256:2, 2] += 1
        return torch.concat((one_ball[0:256:2], one_ball[1:256:2]))
