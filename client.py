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

from argparse import ArgumentParser

import numpy as np
import torch

import flwr as fl

import mnist

from configs import parser

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from model.reconstruct.model_main import ConceptAutoencoder

import os
from configs import parser
from PIL import Image
from utils.draw_tools import draw_bar, draw_plot
import numpy as np
import shutil
from utils.tools import attention_estimation_mnist
from utils.record import apply_colormap_on_image, show
from torch.utils.data import DataLoader, Subset


shutil.rmtree('vis/', ignore_errors=True)       #递归地删除文件
shutil.rmtree('vis_pp/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)
os.makedirs('vis_pp/', exist_ok=True)


DATA_ROOT = "./data/mnist"

if __name__ == "__main__":
    # Training settings

    args = parser.parse_args()



    # Load MNIST data
    train_loader, test_loader = mnist.load_data(
        data_root=DATA_ROOT,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        cid=args.cid,
        nb_clients=args.nb_clients,
    )

    # pylint: disable=no-member
    device = torch.device( "cpu")
    # pylint: enable=no-member
    # Instantiate client
    client = mnist.PytorchMNISTClient(
        cid=args.cid,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=device,
    )


    ip_address = '127.0.0.1'  # here you should write the server ip-address
    server_address=ip_address + ':8080'
    

    # Start client
    fl.client.start_client(server_address, client)
    print('训练完成')



    transform = transforms.Compose([transforms.ToTensor()])
    transform2 = transforms.Normalize((0.1307,), (0.3081,))
    #test_imgs=test_set.data
    val_imgs = datasets.MNIST('./data', train=False, download=True, transform=None).data
    val_target = datasets.MNIST('./data', train=False, download=True, transform=None).targets
    valset = datasets.MNIST('../data', train=False, download=True, transform=transform)


    #test_imgs=test_loader.data



    valloader = DataLoader(valset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers,  pin_memory=False)
    model = ConceptAutoencoder(args, num_concepts=args.num_cpt, vis=True)
    #device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(f"saved_model/mnist_model_cpt{args.num_cpt}.pt", map_location="cpu")  # 模型加载，可用于推理或者是继续训练
    model.load_state_dict(checkpoint, strict=True)  # 加载预训练模型

    model.eval()

    data, label = next(iter(valloader)) # 迭代器，迭代一次获得一个batch的数据 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 0, 17, 26, 34,
    # 9   9
    # 15   5
    # 2   1
    # 3   0
    # 4   4
    # 32   3
    # 21   6
    # 84   8
    # 1    2

    index = args.index

    img_orl = Image.fromarray((data[index][0].cpu().detach().numpy() * 255).astype(np.uint8),
                              mode='L')  # 实现array到image的转换

    img = data[index].unsqueeze(0).to(device)  # unsqueeze() 这个 函数 主要是对数据维度进行扩充。. 给指定位置加上维数为一的维度
    cpt, pred, cons, att_loss, pp = model(transform2(img))
    # print(torch.softmax(pred, dim=-1))
    print("The prediction is: ", torch.argmax(pred, dim=-1))  # dim=n-1=-1

    cons = cons.view(28, 28).cpu().detach().numpy()  # view()的作用相当于numpy中的reshape，重新定义矩阵的形状
    show(data[index].numpy()[0], cons)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)  # 作用：创建一个数组
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
        heatmap_on_image.save("vis/" + str(args.cid)+ f'0_slot_mask_{id}.png')

    # att_record = attention_estimation_mnist(val_imgs, val_target, model, transform, transform2, device, name=7)
    # print(att_record.shape)
    # draw_plot(att_record, "7")

    if args.deactivate == -1:
        is_start = True
        for batch_idx, (data, label) in enumerate(valloader):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            data, label = transform2(data).to(device), label.to(device)
            cpt, pred, out, att_loss, pp = model(data, None, "pass")

            if is_start:
                all_output = cpt.cpu().detach().float()
                all_label = label.unsqueeze(-1).cpu().detach().float()
                is_start = False
            else:
                all_output = torch.cat((all_output, cpt.cpu().detach().float()), 0)
                all_label = torch.cat((all_label, label.unsqueeze(-1).cpu().detach().float()), 0)

        all_output = all_output.numpy().astype("float32")
        all_label = all_label.squeeze(-1).numpy().astype("float32")

        print("cpt visualization")
        for j in range(args.num_cpt):
            root = 'vis_pp/' + str(args.cid)  + "cpt" + str(j + 1) + "/"
            os.makedirs(root, exist_ok=True)
            selected = all_output[:, j]
            ids = np.argsort(-selected, axis=0)  # 返回的是元素值从小到大排序后的索引值的数组  axis:需要排序的维度
            idx = ids[:args.top_samples]
            for i in range(len(idx)):
                img_orl =val_imgs[idx[i]]
                img_orl = Image.fromarray(img_orl.numpy())
                img_orl.save(root +  f'/origin_{i}.png')
                img = transform2(transform(img_orl))
                cpt, pred, out, att_loss, pp = model(img.unsqueeze(0).to(device), ["vis", root], [j, i])
                slot_image = np.array(Image.open(root + f'{i}.png'), dtype=np.uint8)
                heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
                heatmap_on_image.save(root + f'mask_{i}.png')

