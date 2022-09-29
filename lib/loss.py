import torch
from pytorch3d.loss import chamfer_distance as pt3d_cd


def chamfer_distance(p1, p2):
    if not torch.are_deterministic_algorithms_enabled():
        return pt3d_cd(p1, p2)
    s1 = torch.sum(torch.min(torch.cdist(p1, p2, p=2)**2, 2).values, 1)
    s2 = torch.sum(torch.min(torch.cdist(p2, p1, p=2)**2, 2).values, 1)
    return (torch.sum(s1) + torch.sum(s2)) / (p1.shape[-2] * s1.shape[0]), None
