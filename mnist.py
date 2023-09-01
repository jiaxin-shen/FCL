# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PyTorch MNIST image classification.

The code is generally adapted from PyTorch's Basic MNIST Example.
The original code can be inspected in the official PyTorch github:

https://github.com/pytorch/examples/blob/master/mnist/main.py
"""


# mypy: ignore-errors
# pylint: disable=W0223

import timeit
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler,Subset
from torchvision import datasets, transforms

import flwr as fl
from configs import parser

import os

from model.reconstruct.slots import ScouterAttention
from model.reconstruct.position_encode import build_position_encoding


import torch
import torch.nn.functional as F
from model.retrieval.loss import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
#from .record import AverageMeter, ProgressMeter, show
#from .tools import cal_acc, predict_hash_code, mean_average_precision
from utils.record import AverageMeter, ProgressMeter, show
from model.retrieval.loss import batch_cpt_discriminate, att_consistence, quantization_loss, att_area_loss
args = parser.parse_args()

np.set_printoptions(threshold=np.inf)
os.makedirs('saved_model/', exist_ok=True)




def dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    client_id: int,
    number_of_clients: int,
) -> torch.utils.data.DataLoader:
    """Helper function to partition datasets
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to be partitioned into *number_of_clients* subsets.
    batch_size: int
        Size of mini-batches used by the returned DataLoader.
    client_id: int
        Unique integer used for selecting a specific partition.
    number_of_clients: int
        Total number of clients launched during training. This value dictates the number of partitions to be created.
    Returns
    -------
    data_loader: torch.utils.data.Dataset
        DataLoader for specific client_id considering number_of_clients partitions.
    """

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
        dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler
    )
    return data_loader


def load_data(
    data_root: str,
    train_batch_size: int,
    test_batch_size: int,
    cid: int,
    nb_clients: int,
) -> Tuple[DataLoader, DataLoader]:
    """Helper function that loads both training and test datasets for MNIST.
    Parameters
    ----------
    data_root: str
        Directory where MNIST dataset will be stored.
    train_batch_size: int
        Mini-batch size for training set.
    test_batch_size: int
        Mini-batch size for test set.
    cid: int
        Client ID used to select a specific partition.
    nb_clients: int
        Total number of clients launched during training. This value dictates the number of unique to be created.
    Returns
    -------
    (train_loader, test_loader): Tuple[DataLoader, DataLoader]
        Tuple contaning DataLoaders for training and test sets.
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    # Create partitioned datasets based on the total number of clients and client_id
    train_loader = dataset_partitioner(
        dataset=train_dataset,
        batch_size=train_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
    )

    test_loader = dataset_partitioner(
        dataset=test_dataset,
        batch_size=test_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
    )

    return (train_loader, test_loader )


class ConceptAutoencoder(nn.Module):
    def __init__(self, args, num_concepts, vis=False):
        super(ConceptAutoencoder, self).__init__()
        hidden_dim = 32
        self.args = args
        self.num_concepts = num_concepts
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=2, padding=1)  # b, 16, 10, 10
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
        self.slots = ScouterAttention(hidden_dim, num_concepts, vis=self.vis)
        self.aggregate = Aggregate(args, num_concepts)

    def forward(self, x, loc=None, index=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        pe = self.position_emb(x)
        x_pe = x + pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        updates, attn = self.slots(x_pe, x, loc, index)
        cpt_activation = attn
        attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)

        x = attn_cls.reshape(b, -1)
        cpt = self.activation(x)  #t
        x = cpt
        if self.args.deactivate != -1:
            x[0][self.args.deactivate-1] = 0
        pred = self.aggregate(x)
        x = self.relu(self.fc1(x))
        x = self.tan(self.fc2(x))
        return (cpt - 0.5) * 2, pred, x, attn, updates


class Aggregate(nn.Module):
    def __init__(self, args, num_concepts):
        super(Aggregate, self).__init__()
        self.args = args
        if args.layer != 1:
            self.fc1 = nn.Linear(num_concepts, num_concepts)
        self.fc2 = nn.Linear(num_concepts, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.args.layer != 1:
            x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def cal_acc(preds, labels):
    with torch.no_grad():
        pred = preds.argmax(dim=-1) #argmax返回的是最大数的索引.argmax有一个参数axis,默认是0,表示第几维的最大值.看二维的情况.
        acc = torch.eq(pred, labels).sum().float().item() / labels.size(0)
        return acc


def train(args, model, device, loader, rec_loss ,epochs):
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    #reconstruction_loss = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print(f"Training {epochs} epoch(s) w/ {len(loader)} mini-batches each")
    for epoch in range(epochs):
        num_examples_train: int = 0
        recon_losses = AverageMeter('Reconstruction Loss', ':.4')
        # att_losses = AverageMeter('Att Loss', ':.4')
        pred_losses = AverageMeter('Pred Loss', ':.4')
        batch_dis_losses = AverageMeter('Dis_loss_batch', ':.4')
        consistence_losses = AverageMeter('Consistence_loss', ':.4')
        q_losses = AverageMeter('Q_loss', ':.4')
        pred_acces = AverageMeter('Acc', ':.4')
        progress = ProgressMeter(len(loader),
                                [recon_losses, pred_losses, pred_acces, batch_dis_losses, consistence_losses, q_losses],
                                #[ recon_losses,pred_losses, pred_acces ,batch_dis_losses, q_losses],
                                prefix="Epoch: [{}]".format(epoch))

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            num_examples_train += len(data)
            cpt, pred, out, att, update = model(data)

            loss_pred = F.nll_loss(F.log_softmax(pred, dim=1), label)
            acc = cal_acc(pred, label)
            reconstruction_loss = rec_loss(out.view(data.size(0), 1, 28, 28), data)
            quantity_loss = quantization_loss(cpt)
            batch_dis_loss = batch_cpt_discriminate(update, att)
            consistence_loss = att_consistence(update, att)
            att_loss = att_area_loss(att)  # attention loss used to prevent overflow

            recon_losses.update(reconstruction_loss.item())
            # att_losses.update(att_loss.item())
            pred_losses.update(loss_pred.item())
            pred_acces.update(acc)
            q_losses.update(quantity_loss.item())
            batch_dis_losses.update(batch_dis_loss.item())
            consistence_losses.update(consistence_loss.item())
            '''
            loss_total =  args.weak_supervision_bias * reconstruction_loss + args.att_bias * att_loss + loss_pred + args.quantity_bias * quantity_loss + \
                         args.distinctiveness_bias * batch_dis_loss

            '''
            loss_total = args.weak_supervision_bias * reconstruction_loss + args.att_bias * att_loss + loss_pred + args.quantity_bias * quantity_loss + \
                        args.distinctiveness_bias * batch_dis_loss + args.consistence_bias * consistence_loss
            


            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                progress.display(batch_idx)
        scheduler.step()
    return num_examples_train #这个客户端拥有的样本数量

@torch.no_grad()
def test(model, loader , device):
    model.eval()
    record_res = 0.0
    record_att = 0.0
    accs = 0
    L = len(loader)
    rec_loss = nn.MSELoss()
    num_test_samples: int = 0

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        num_test_samples += len(data)
        cpt, pred, out, att, update = model(data)

        acc = cal_acc(pred, label)
        reconstruction_loss = rec_loss(out.view(data.size(0), 28, 28), data)
        record_res += reconstruction_loss.item()
        att_loss = att_area_loss(att)
        record_att += att_loss.item()
        accs += acc
    return num_test_samples, round(record_res/L, 4), round(record_att/L, 4), round(accs/L, 4)


def vis_one(model, device, loader, epoch=None, select_index=0):
    data, label = iter(loader).next()
    img_orl = data[select_index]
    img = img_orl.unsqueeze(0).to(device)
    pred = model(img)[2].view(28, 28).cpu().detach().numpy()
    show(img_orl.numpy()[0], pred, epoch)

#print(args.num_cpt)
#model = ConceptAutoencoder(args, num_concepts=args.num_cpt,vis=False)
#params = [p for p in model.parameters() if p.requires_grad] #p需要梯
#optimizer = torch.optim.AdamW(params, lr=args.lr)
reconstruction_loss = nn.MSELoss()

record_res = []
record_att = []
accs = []



class   PytorchMNISTClient(fl.client.Client):
    """Flower client implementing MNIST handwritten classification using PyTorch."""

    def __init__(
        self,
        cid: int,
        train_loader: datasets,
        test_loader: datasets,
        epochs: int,
        device: torch.device = torch.device("cpu"),

    ) -> None:

        self.model = ConceptAutoencoder(args, num_concepts=args.num_cpt,vis=False).to(device)
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
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
        #保存模型参数

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
        #print(parameters)
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

        num_examples_train: int = train(args, self.model, self.device, self.train_loader, reconstruction_loss, epochs=self.epochs)

        parm = {}
        for name, parameters in self.model.named_parameters():
            print(self.cid, name, ':', parameters.size())
            parm[name] = parameters.cpu().detach().numpy()
    

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.Weights = self.get_weights()

        #-----------------------模拟恶意客户端-------------------------------------
        if self.cid==1:
            # print("eeeeeeeeeeeeeeeeeee",weights_prime[-2][0])
            for a in range(len(weights_prime[-2])):
                for b in range(len(weights_prime[-2][a])):
                    weights_prime[-2][a][b]=weights_prime[-2][a][b]*-1
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
        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.set_weights(weights)

        (
            num_test_samples,
            record_res,
            record_att,
            accs,
        ) = test(self.model, self.test_loader, device=self.device)
        print(
            f"Client {self.cid} - Evaluate on {num_test_samples} samples: Average loss: {record_res:.4f}, Accuracy: {100*accs:.2f}%\n"
        )
        torch.save(self.model.state_dict(), f"saved_model/mnist_model_cpt{args.num_cpt}.pt")

        #df = pd.DataFrame(columns=[ 'cid', 'average Loss', ' accuracy'])  # 列名
        #df.to_csv("D:\\pytorch\\flower\\flower-main\\src\\py\\flwr_example\\quickstart_pytorch\\test_acc.csv", index=False)  # 路径可以根据需要更改
        #list=[self.cid,test_loss,100*accuracy]

        #data = pd.DataFrame([list])
        #data.to_csv('D:\\pytorch\\flower\\flower-main\\src\\py\\flwr_example\\quickstart_pytorch\\test_acc.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

        # Return the number of evaluation examples and the evaluation result (loss)
        return fl.common.EvaluateRes(
            loss=float(record_res),
            num_examples=num_test_samples,
            accuracy=float(accs),
        )



