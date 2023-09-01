
import struct
import timeit
from collections import OrderedDict
from typing import Tuple

import numpy as np

import torch.nn as nn

from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler,Subset
from torchvision import datasets, transforms

import flwr as fl
from configs import parser
import sys
import os




from timm.models import create_model

from model.retrieval.slots import ScouterAttention, vis
from model.retrieval.position_encode import build_position_encoding
from timm.models.layers import SelectAdaptivePool2d

from model.retrieval.loss import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
from utils.record import AverageMeter, ProgressMeter, show
from utils.tools import cal_acc, predict_hash_code, mean_average_precision


import torch

import torch.nn.functional as F

from model.retrieval.loss import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
#from .record import AverageMeter, ProgressMeter, show
#from .tools import cal_acc, predict_hash_code, mean_average_precision
from utils.record import AverageMeter, ProgressMeter, show
from model.retrieval.loss import batch_cpt_discriminate, att_consistence, quantization_loss, att_area_loss

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from loaders.CUB200 import CUB_200
from loaders.ImageNet import ImageNet

from PIL import Image




args = parser.parse_args()

np.set_printoptions(threshold=np.inf)
os.makedirs('saved_model/', exist_ok=True)

#data_dir = '/home/coder/projects/botcl-fed/botcl-fed/data/image/tiny-imagenet-200'



def train1_dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    client_id: int,
    number_of_clients: int,
) -> torch.utils.data.DataLoader:
   

    # Set the seed so we are sure to generate the same global batches
    # indices across all clients
    np.random.seed(123)

    # Get the data corresponding to this client
    dataset_size = len(dataset)
    print('1', dataset_size)
    
    nb_samples_per_clients = dataset_size // number_of_clients
    print('2',nb_samples_per_clients)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    # Get starting and ending indices w.r.t CLIENT_ID
    start_ind = client_id * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    print(client_id,'start',start_ind )
    print(client_id, 'end', end_ind)
    data_sampler = SubsetRandomSampler(dataset_indices[start_ind:end_ind])
    data_set= Subset(dataset,dataset_indices[start_ind:end_ind])
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=False, drop_last=True
    )


    return data_loader


def dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    client_id: int,
    number_of_clients: int,
) -> torch.utils.data.DataLoader:
    
    # Set the seed so we are sure to generate the same global batches
    # indices across all clients
    np.random.seed(123)

    # Get the data corresponding to this client
    dataset_size = len(dataset)
    
    nb_samples_per_clients = dataset_size // number_of_clients
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    # Get starting and ending indices w.r.t CLIENT_ID
    start_ind = client_id * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    data_sampler = SubsetRandomSampler(dataset_indices[start_ind:end_ind])
    data_set= Subset(dataset,dataset_indices[start_ind:end_ind])
    data_loader = torch.utils.data.DataLoader(
        data_set, batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=False, drop_last=False
    )
    return data_loader


def get_train_transformations(args, norm_value):
    aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)

def get_val_transformations(args, norm_value):
    aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)


def get_transform(args):
    #imagenet cub200
    if args.dataset == "CUB200":
        transform_train = get_train_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform_val = get_val_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return {"train": transform_train, "val": transform_val}
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), transforms.ToTensor(),
                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        return {"train": transform, "val": transform}





def select_dataset(args, transform):
    #imagenet cub200
    if args.dataset == 'CUB200' :
        dataset_train = CUB_200(args, train=True, transform=transform["train"])
        dataset_val = CUB_200(args, train=False, transform=transform["val"])
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('./datacifar10',  train=True,  download=True,transform=transform["train"])
        dataset_val =  datasets.CIFAR10('./datacifar10',  train=False, download=True,transform=transform["train"])
    return dataset_train, dataset_val



def load_data(

    train_batch_size: int,
    test_batch_size: int,
    cid: int,
    nb_clients: int,
) -> Tuple[DataLoader, DataLoader]:
 
    transform = get_transform(args)

    #train_dataset = datasets.ImageFolder(root='/home/coder/projects/botcl-fed/botcl-fed/data/ImageNet/train',transform=transform["train"])

    #val_dataset = datasets.ImageFolder(root='/home/coder/projects/botcl-fed/botcl-fed/data/ImageNet/val',transform=transform["val"])

    train_dataset, val_dataset = select_dataset(args, transform)
    print('训练集',len(train_dataset))
    print('验证集',len(val_dataset))

    #train_dataset = TinyImageNet(data_dir, train=True, transform=get_train_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]))
    #val_dataset = TinyImageNet(data_dir, train=False,transform = get_val_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]))

    # Create partitioned datasets based on the total number of clients and client_id
    train_loader1 = train1_dataset_partitioner(
        dataset=train_dataset,
        batch_size=train_batch_size,
        #shuffle=True,
        client_id=cid,
        number_of_clients=nb_clients,
        #pin_memory=False, drop_last=True
    )
    train_loader2 = dataset_partitioner(
        dataset=train_dataset,
        batch_size=train_batch_size,
        #shuffle=False,
        client_id=cid,
        number_of_clients=nb_clients,
        #pin_memory=False, drop_last=False
    )

    val_loader = dataset_partitioner(
        dataset=val_dataset,
        batch_size=test_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
        #shuffle=False,
        #pin_memory=False, drop_last=False
    )

    return (train_loader1, train_loader2, val_loader )


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
            # self.featuremap1 = x
            x = self.norm(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            #self.featuremap2 = pe
            x_pe = x + pe
            #self.featuremap3 = x_pe.detach()

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            updates, attn = self.slots(x_pe, x, weight, things)
            #print("A",attn)
            #self.featuremap4 = attn.detach()
            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            #print("attn_cls",attn_cls)
            cpt = self.activation(attn_cls) #t
            #print("cpt#############################",cpt)
            
            cls = self.cls(cpt)
            #print("cls",cls)
            
            return (cpt - 0.5) * 2, cls, attn, updates
        else:
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            return x

    def get_featuremap(self):
        return self.featuremap1

class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)

    if args.dataset == "MNIST":
            bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
    bone.global_pool = Identical()
    bone.fc = Identical()
    # fix_parameter(bone, [""], mode="fix")
    # fix_parameter(bone, ["layer4", "layer3"], mode="open")
    return bone


def train(args, model, device, loader, epochs):
    model.train()


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    #reconstruction_loss = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print(f"Training {epochs} epoch(s) w/ {len(loader)} mini-batches each")
    for epoch in range(epochs):
        num_examples_train: int = 0
        retri_losses = AverageMeter('Retri_loss Loss', ':.4')
        # att_losses = AverageMeter('Att Loss', ':.4')
        q_losses = AverageMeter('Q_loss', ':.4')
        batch_dis_losses = AverageMeter('Dis_loss_batch', ':.4')
        consistence_losses = AverageMeter('Consistence_loss', ':.4')
        pred_acces = AverageMeter('Acc', ':.4')
        if not args.pre_train:
            show_items = [retri_losses, q_losses, pred_acces, batch_dis_losses, consistence_losses]
        else:
            show_items = [pred_acces]
        progress = ProgressMeter(len(loader), show_items, prefix="Epoch: [{}]".format(epoch))

        for batch_idx, (data, label) in enumerate(loader):

            data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
            num_examples_train += len(data)
            if not args.pre_train:
                cpt, pred, att, update = model(data)
                pred = F.log_softmax(pred, dim=-1)
                retri_loss, quantity_loss = get_retrieval_loss(cpt, label, args.num_classes, device)
                loss_pred = F.nll_loss(pred, label)
                acc = cal_acc(pred, label, False)
                batch_dis_loss = batch_cpt_discriminate(update, att)
                consistence_loss = att_consistence(update, att)
                attn_loss = att_area_loss(att)

                retri_losses.update(retri_loss.item())
                # att_losses.update(attn_loss.item())
                q_losses.update(quantity_loss.item())
                batch_dis_losses.update(batch_dis_loss.item())
                consistence_losses.update(consistence_loss.item())
                pred_acces.update(acc)

                loss_total = args.weak_supervision_bias * retri_loss + args.att_bias * attn_loss + args.quantity_bias * quantity_loss + \
                             loss_pred - args.consistence_bias * consistence_loss + args.distinctiveness_bias * batch_dis_loss
            else:
                cpt = F.log_softmax(model(data), dim=-1)
                retri_loss = F.nll_loss(cpt, label)
                #print('cpt',cpt)
                #print('label',label)
                acc = cal_acc(cpt, label, False)
                pred_acces.update(acc)
                loss_total = retri_loss

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                progress.display(batch_idx)

        scheduler.step()
    return num_examples_train #这个客户端拥有的样本数量

def test_MAP(args, model, database_loader, test_loader, device):
    model.eval()
    print('Waiting for generate the hash code from database')
    database_hash, database_labels, database_acc = predict_hash_code(args, model, database_loader, device)
    print('Waiting for generate the hash code from test set')
    test_hash, test_labels, test_acc = predict_hash_code(args, model, test_loader, device)
    print("label", database_labels.shape)
    print('Calculate MAP.....')

    if args.num_classes > 400:
        print("class number over" + str(args.num_classes) + "not do retrieval evaluation during training due to the time cost. Set MAP as 0.")
        MAP = 0
    else:
        MAP, R, APx = mean_average_precision(args, database_hash, test_hash, database_labels, test_labels)
    # print(MAP)
    # print(R)
    # print(APx)
    return MAP, test_acc

@torch.no_grad()
def test( model, test_loader, device):
    model.eval()
    record = 0.0
    loss=0.0
    num_test_samples: int = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        num_test_samples += len(data)
        #cpt = F.log_softmax(model(data), dim=-1)
        cpt, pred, att, update = model(data)

        #retri_loss, quantity_loss = get_retrieval_loss(cpt, label, args.num_classes, device)
        #loss_pred = F.nll_loss(pred, label)

        #########cpt = model(data)
        pred = F.log_softmax(pred, dim=-1)
        retri_loss = F.nll_loss(pred, label)
        acc = cal_acc(pred, label, False)

        ########retri_loss = F.nll_loss(cpt, label)
        ########acc = cal_acc(cpt, label, False)
        record += acc
        loss += retri_loss.item()
    print("ACC:", record/len(test_loader))
    return num_test_samples, loss/len(test_loader), record/len(test_loader)



#print(args.num_cpt)
#model = ConceptAutoencoder(args, num_concepts=args.num_cpt,vis=False)
#params = [p for p in model.parameters() if p.requires_grad] #p需要梯
#optimizer = torch.optim.AdamW(params, lr=args.lr)
reconstruction_loss = nn.MSELoss()

record_res = []
record_att = []
accs = []




class PytorchClient(fl.client.Client):
    """Flower client implementing MNIST handwritten classification using PyTorch."""

    def __init__(
            self,
            cid: int,
            train_loader1: datasets,
            train_loader2: datasets,
            val_loader: datasets,
            epochs: int,
            device: torch.device = torch.device("cuda:0"),

    ) -> None:
        self.model = MainModel(args).to(device)
        self.cid = cid
        self.train_loader1 = train_loader1
        self.train_loader2 = train_loader2
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays.

        Parameters
        ----------
        weights: fl.common.Weights
            Weights received by the server and set to local model


        Returns
        -------

        """
        # 保存模型参数

        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> fl.common.ParametersRes:
        """Encapsulates the weights into Flower Parameters."""
        weights: fl.common.Weights = self.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        # print(parameters)
        return fl.common.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        """Trains the model on local dataset

        Parameters
        ----------
        ins: fl.common.FitIns
           Parameters sent by the server to be used during training.

        Returns
        -------
            Set of variables containing the new set of weights and information the client.

        """

        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        fit_begin = timeit.default_timer()

        # Set model parameters/weights
        self.set_weights(weights)

        # Train model

        num_examples_train: int = train(args, self.model, self.device, self.train_loader1,  epochs=self.epochs)

        parm = {}
        print("*"*30)
        for name in self.model.named_children():
            print(name[0])
        print("*"*30)
        for name, parameters in self.model.named_parameters():
            print(self.cid, name, ':', parameters.size())
            parm[name] = parameters.detach().cpu().numpy()

        #exit()

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.Weights = self.get_weights()

        #-----------------------模拟恶意客户端-------------------------------------
        if self.cid==1:
            print("eeeeeeeeeeeeeeeeeee",len(weights_prime[-2]))
            print("fffffffffffffffffff",weights_prime[-2][0])
            for a in range(len(weights_prime[-2])):
                for b in range(len(weights_prime[-2][a])):
                    weights_prime[-2][a][b]=weights_prime[-2][a][b]*-1
            print("gggggggggggggggggg",weights_prime[-2][0])
        #-----------------------模拟恶意客户端-------------------------------------           
            

        params_prime = fl.common.weights_to_parameters(weights_prime)
        

        fit_duration = timeit.default_timer() - fit_begin
        
        
        
        
        
        return fl.common.FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )


    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        """
        Parameters
        ----------
        ins: fl.common.EvaluateIns
           Parameters sent by the server to be used during testing.


        Returns
        -------
            Information the clients testing results.

        """
        map, acc = test_MAP(args,self.model, self.train_loader2, self.val_loader, self.device)
        print("test_ACC: ", acc)
        print("MAP", map)



        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.set_weights(weights)

        (
            num_test_samples,
            loss,
            accs,
        ) = test(self.model, self.val_loader, device=self.device)
        print(
            f"Client {self.cid} - Evaluate on {num_test_samples} samples: Average loss: {loss:.4f}, Accuracy: {100*accs:.2f}%\n"
        )
        torch.save(self.model.state_dict(), os.path.join(args.output_dir,
                                                    f"{args.dataset}_{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                                                    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"))

       
        return fl.common.EvaluateRes(
            loss=map,
            num_examples=num_test_samples,
            accuracy=float(acc),
        )

