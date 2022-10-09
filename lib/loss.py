import torch
from pytorch3d.loss import chamfer_distance as pt3d_cd
import torch
from torch import nn
from torch.autograd import Function
import emd


def chamfer_distance(p1, p2):
    if not torch.are_deterministic_algorithms_enabled():
        return pt3d_cd(p1, p2)
    s1 = torch.sum(torch.min(torch.cdist(p1, p2, p=2)**2, 2).values, 1)
    s2 = torch.sum(torch.min(torch.cdist(p2, p1, p=2)**2, 2).values, 1)
    return (torch.sum(s1) + torch.sum(s2)) / (p1.shape[-2] * s1.shape[0]), None

class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m)
        assert(xyz1.size()[0] == xyz2.size()[0])
        assert(n % 1024 == 0)
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)

        loss = torch.sqrt(dist).mean()
        # return dist, assignment
        return loss, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None

class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()
        self.train_param(True)

    def forward(self, input1, input2, eps=None, iters=None):
        if eps is None:
            eps = self.eps
        if iters is None:
            iters = self.iters
        return emdFunction.apply(input1, input2, eps, iters)
    
    def train_param(self, mode: bool = True):
        if mode is True:
            self.eps = 0.05
            self.iters = 50
        else:
            self.eps = 0.002
            self.iters = 10000
        