from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.retrieval.slots import ScouterAttention, vis
from model.retrieval.position_encode import build_position_encoding
from timm.models.layers import SelectAdaptivePool2d


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)

    bone.global_pool = Identical()
    bone.fc = Identical()
    # fix_parameter(bone, [""], mode="fix")
    # fix_parameter(bone, ["layer4", "layer3"], mode="open")
    return bone


class MainModel(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        hidden_dim = 128
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis

        if not self.pre_train:
            self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
            self.slots = ScouterAttention(args, hidden_dim, num_concepts, vis=self.vis)
            self.norm = nn.BatchNorm2d(hidden_dim)
            self.scale = 1
            self.cls = torch.nn.Linear(num_concepts, num_classes)
        else:
            self.fc = nn.Linear(self.num_features, args.num_classes)
            self.drop_rate = 0

    def forward(self, x, weight=None, things=None):
        x = self.back_bone(x)
        # x = x.view(x.size(0), self.num_features, self.feature_size, self.feature_size)
        if not self.pre_train:
            x = self.conv1x1(x)
            self.featuremap1 = x
            x = self.norm(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            self.featuremap2 = pe
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            updates, attn = self.slots(x_pe, x, weight, things)
            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            cpt = self.activation(attn_cls)
            cls = self.cls(cpt)
            return (cpt - 0.5) * 2, cls, attn, updates
        else:
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            return x


# if __name__ == '__main__':
#     model = MainModel()
#     inp = torch.rand((2, 1, 224, 224))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)


