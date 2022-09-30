import torch
from torch import nn


class Pixel2Point(nn.Module):
    def __init__(self, initial_point=0):
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
        if initial_point == 2:
            self.initial_point = self.multiple_sphere()
        elif initial_point == 1:
            self.initial_point = self.fibonacci_sphere()
        else:
            self.initial_point = self.fibonacci_sphere()

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

    def sphere(self, offset=(0, 0, 0)):
        u = torch.linspace(0, 2 * torch.pi, 16)
        v = torch.linspace(0, torch.pi, 16)
        x = torch.flatten(1 * torch.outer(torch.cos(u), torch.sin(v)))
        y = torch.flatten(1 * torch.outer(torch.sin(u), torch.sin(v)))
        z = torch.flatten(1 * torch.outer(torch.ones(torch.Tensor.size(u)), torch.cos(v)))
        return torch.stack((x + offset[0], y + offset[1], z + offset[2]), 1)

    def fibonacci_sphere(self, num_pts=256, offset=(0, 0, 0), radius=1):
        # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        indices = torch.arange(0, num_pts, dtype=torch.float32) + 0.5
        phi = torch.arccos(1 - 2*indices/num_pts)
        theta = torch.pi * (1 + 5**0.5) * indices
        x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
        return torch.stack((x * radius + offset[0], y * radius + offset[1], z * radius + offset[2]), 1)

    def multiple_sphere(self):
        sphere_1 = self.fibonacci_sphere(128, (0, 0, 0), 1)
        sphere_2 = self.fibonacci_sphere(128, (2, 2, 2), 1)
        return torch.cat((sphere_1, sphere_2), 0)
