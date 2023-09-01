import torch.nn as nn
import torch
from model.reconstruct.slots import ScouterAttention
from model.reconstruct.position_encode import build_position_encoding


class ConceptAutoencoder(nn.Module):
    def __init__(self, args, num_concepts, vis=False):
        super(ConceptAutoencoder, self).__init__()
        hidden_dim = 32
        self.args = args
        self.num_concepts = num_concepts
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=2, padding=1)  # b, 16, 10, 10  batchsize channels 长 宽
        self.conv2 = nn.Conv2d(16, hidden_dim, (3, 3), stride=2, padding=1)  # b, 8, 3, 3
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_concepts, 400)
        self.fc2 = nn.Linear(400, 28 * 28)
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.vis = vis
        self.scale = 1
        self.activation = nn.Tanh()
        self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
        self.slots = ScouterAttention(hidden_dim, num_concepts, vis=self.vis)# 抽取概念的模块   两个输入，一个是隐藏层的维度，一个是概念的数量
        self.aggregate = Aggregate(args, num_concepts)

    def forward(self, x, loc=None, index=None): #x=data
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))# x=F 是特征图

        pe = self.position_emb(x) #为了保留空间信息，将位置嵌入P添加到特征地图中，即F′=F+P。
        x_pe = x + pe  # F′=x_pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))  #这里把长和宽乘起来了
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        updates, attn = self.slots(x_pe, x, loc, index) #attention表示每个concept激活的面积
        cpt_activation = attn
        attn_cls = self.scale * torch.sum(cpt_activation, dim=-1) #加起来就表示A1l 就是将attention 的空间维度加和

        x = attn_cls.reshape(b, -1) # -1表示列数自动计算   因为上面把batchsize 算进去了 ，所以现在把他弄出来
        cpt = self.activation(x) #cpt是一个0到1之间的值   cpt =概念激活t
        x = cpt    #x就代表concept的激活或者没激活
        if self.args.deactivate != -1:
            x[0][self.args.deactivate-1] = 0  #某个concept的不激活，把他变成0
        pred = self.aggregate(x)
        x = self.relu(self.fc1(x))
        x = self.tan(self.fc2(x))
        return (cpt - 0.5) * 2, pred, x, attn, updates #concept原来是0到1 减了0.5*2变成-1到1  论文里要这样计算  pred是判断它是哪一类，x是reconstructed 图片  ，update对recon没有意义  可以不用管
        #cpt 是k维的[0,1]向量   pred 是判断它是哪一类  x 是啥？？

class Aggregate(nn.Module):
    def __init__(self, args, num_concepts):
        super(Aggregate, self).__init__()
        self.args = args
        if args.layer != 1:
            self.fc1 = nn.Linear(num_concepts, num_concepts)
        self.fc2 = nn.Linear(num_concepts, 10)  #权重就是分数

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.args.layer != 1:
            x = self.relu(self.fc1(x))
        x = self.fc2(x)  #通过fc来判断到底是哪一类   这里输入的x 是权重矩阵
        #print('全连接层的权重形状', self.fc2.weight.shape)
        return x


# class ConceptAutoencoder(nn.Module):
#     def __init__(self, num_concepts):
#         super(ConceptAutoencoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5))
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5))
#         self.fc1 = nn.Linear(20 * 20 * 20, 16)
#         self.fc2 = nn.Linear(16, num_concepts)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.decoder = nn.Sequential(
#             nn.Linear(num_concepts, 16), nn.ReLU(True),
#             nn.Linear(16, 64), nn.ReLU(True),
#             nn.Linear(64, 128), nn.ReLU(True),
#             nn.Linear(128, 28 * 28),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         cpt = self.relu(self.conv1(x))
#         cpt = self.relu(self.conv2(cpt))
#         b = cpt.size()[0]
#         cpt = cpt.view(b, -1)
#         cpt = self.relu(self.fc1(cpt))
#         encoder = self.fc2(cpt)
#         decoder = self.decoder(encoder)
#         return decoder


# class ConceptAutoencoder(nn.Module):
#     def __init__(self, num_concepts):
#         super(ConceptAutoencoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=2, padding=1)  # b, 16, 10, 10
#         self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1)  # b, 8, 3, 3
#
#         self.dconv1 = nn.ConvTranspose2d(32, 16, (2, 2), stride=2)  # b, 16, 5, 5
#         self.dconv2 = nn.ConvTranspose2d(16, 1, (2, 2), stride=2)  # b, 16, 5, 5
#
#         self.relu = nn.ReLU(inplace=True)
#         self.tan = nn.Tanh()
#
#     def forward(self, x):
#         cpt = self.relu(self.conv1(x))
#         cpt = self.relu(self.conv2(cpt))
#         cpt = self.relu(self.dconv1(cpt))
#         cpt = self.tan(self.dconv2(cpt))
#         return cpt


# if __name__ == '__main__':
#     model = ConceptAutoencoder(num_concepts=10)
#     inp = torch.rand((2, 1, 28, 28))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)