from torch import nn
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np


class ScouterAttention(nn.Module):
    def __init__(self, dim, num_concept, iters=1, eps=1e-8, vis=False, power=1, to_k_layer=3): # dim=32
        super().__init__()
        self.num_slots = num_concept
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # random seed init
        slots_mu = nn.Parameter(torch.randn(1, 1, dim)) #随机初始化 (1,1,32)
        slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim))) #随机初始化  abs : 回输入参数的绝对值作为输出
        mu = slots_mu.expand(1, self.num_slots, -1) #(1,10,32)
        sigma = slots_sigma.expand(1, self.num_slots, -1)
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma)) #mu 和 sigma 用来初始化 slots

        # K layer init
        to_k_layer_list = [nn.Linear(dim, dim)]
        for to_k_layer_id in range(1, to_k_layer):
            to_k_layer_list.append(nn.ReLU(inplace=True))
            to_k_layer_list.append(nn.Linear(dim, dim))
        self.to_k = nn.Sequential(
            *to_k_layer_list
        )

        self.vis = vis
        self.power = power

    def forward(self, inputs, inputs_x, loc, index): #updates, attn = self.slots(x_pe, x, loc, index)
        b, n, d = inputs.shape   # F'
        slots = self.initial_slots.expand(b, -1, -1)
        k, v = self.to_k(inputs), inputs  #K(F')  F'做非线性变化
        for _ in range(self.iters):
            q = slots

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale    #q,k乘积  q是初始化的概念矩阵，k是f'
            dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2, 0, 1])).permute([1, 2, 0])) * \
                   dots.sum(2).sum(1).expand_as(dots.permute([1, 2, 0])).permute([2, 0, 1])

            attn1 = dots.softmax(dim=1) #1这个维度是每个概念
            attn2 = dots.sigmoid()
            attn = attn1 * attn2

            updates = torch.einsum('bjd,bij->bid', inputs_x, attn) #求和
            updates = updates / inputs_x.size(2)

        if self.vis or index is not None:
            if index != "pass":
                slots_vis_raw = attn.clone()
                vis(slots_vis_raw, loc, 7, index)
        return updates, attn  #attention表示每个concept激活的面积


def vis(slots_vis_raw, loc, size, index):
    b = slots_vis_raw.size()[0]
    if loc is not None:
        loc1, loc2 = loc
    else:
        loc1 = "vis"

    for i in range(b):
        slots_vis = slots_vis_raw[i]
        slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
            slots_vis.shape[:1] + (int(size), int(size)))

        slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
        for id, image in enumerate(slots_vis):
            image = Image.fromarray(image, mode='L').resize([28, 28], resample=Image.BILINEAR)
            if index is not None:
                if id == index[0]:
                    image.save(loc2 + f'{index[1]}.png')
                    break
                else:
                    continue
            image.save(f'{loc1}/{i}_slot_{id:d}.png')