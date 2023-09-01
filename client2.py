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
# parser.add_argument('--weak_supervision_bias', type=float, default=0.1, help='weight fot the weak supervision branch')
# parser.add_argument('--att_bias', type=float, default=0.5, help='used to prevent overflow, default as 0.1')#2.调大点或者调小点 0.01或者0.5
# parser.add_argument('--quantity_bias', type=float, default=0.05, help='force each concept to be binary')#3.调小点 0.01
# parser.add_argument('--distinctiveness_bias', type=float, default=0.01, help='refer to paper')#4.设置为0
# parser.add_argument('--consistence_bias', type=float, default=0.1, help='refer to paper')#4.设置为0
# ==============================================================================
from argparse import ArgumentParser

import numpy as np
import torch
#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark =True


import torchvision

import flwr as fl

import mnist2

from configs import parser
import os

import cv2

import matplotlib.pyplot as plt
from termcolor import colored

from utils.engine_retri import train, test_MAP, test
from model.retrieval.model_main import MainModel
from loaders.get_loader import loader_generation
from utils.tools import fix_parameter, print_param
from utils.tools import predict_hash_code, mean_average_precision
from torchvision import datasets, transforms
from model.retrieval.model_main import MainModel
from configs import parser


from PIL import Image

from utils.tools import apply_colormap_on_image
from loaders.get_loader import load_all_imgs, get_transform
from utils.tools import for_retrival, attention_estimation
import h5py
from utils.draw_tools import draw_bar, draw_plot
import shutil
from utils.tools import crop_center
from model.retrieval.position_encode import build_position_encoding
from model.retrieval.slots import ScouterAttention


from torchsummary import summary
import pylab

shutil.rmtree('vis/', ignore_errors=True)
shutil.rmtree('vis_pp/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)
os.makedirs('vis_pp/', exist_ok=True)
np.set_printoptions(suppress=True)

os.makedirs('saved_model/', exist_ok=True)

DATA_ROOT = "./data/image"
IMAGE_FOLDER = './save_image'




def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation_a[name] = output.detach()
    return hook

class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

def feature_imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.detach().numpy().transpose((1, 2, 0)) # 将通道数放在最后一维
    #print("inp",inp)
    inp = np.clip(inp, 0, 1) 
    print("4")
    plt.imshow(inp)
    
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('tuxiang/savefig_example.jpg')
    print("5")
    pylab.show()




if __name__ == "__main__":
    # Training settings


   
    args = parser.parse_args()
    model = MainModel(args)
    device = torch.device("cuda:0")
    '''
    checkpoint = torch.load(os.path.join(args.output_dir,f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"), map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    fix_parameter(model, ["layer1", "layer2", "back_bone.conv1", "back_bone.bn1"], mode="fix")
    print(colored('trainable parameter name: ', "blue"))
    print_param(model)
    print("load pre-trained model finished, start training")
    '''
    if not args.pre_train:
        checkpoint = torch.load(os.path.join(args.output_dir,
                                             f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"),map_location=device)
        #checkpoint = torch.load(os.path.join(args.output_dir,
            #f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt20_use_slot_att.pt"), map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        fix_parameter(model, ["layer1", "layer2", "back_bone.conv1", "back_bone.bn1"], mode="fix")
        print(colored('trainable parameter name: ', "blue"))
        print_param(model)
        print("load pre-trained model finished, start training")
    else:
        print("start training the backbone")



    # Load MNIST data
    train_loader1,train_loader2, val_loader = mnist2.load_data(

        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        cid=args.cid,
        nb_clients=args.nb_clients,
    )


    print("正式训练")
    client = mnist2.PytorchClient(
        cid=args.cid,
        train_loader1=train_loader1,
        train_loader2=train_loader2,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
    )




    ip_address = '127.0.0.1'  # here you should write the server ip-address
    server_address=ip_address + ':8080'
    

    # Start client
    fl.client.start_client(server_address, client)
    print('训练完成')

    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    print("All category:")
    print(cat)
    transform = get_transform(args)["val"]

    # load model and weights
    model = MainModel(args,vis=True)
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()



    #process

    args.process = True
    print('Waiting for generating from database')
    database_hash, database_labels, database_acc = predict_hash_code(args, model, train_loader2, device)
    print('Waiting for generating from test set')
    test_hash, test_labels, test_acc = predict_hash_code(args, model, val_loader, device)

    f = h5py.File(f"data_map/{args.cid}_{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", "w")
    d1 = f.create_dataset("database_hash", data=database_hash)
    d2 = f.create_dataset("database_labels", data=database_labels)
    d6 = f.create_dataset("test_hash", data=test_hash)
    d7 = f.create_dataset("test_labels", data=test_labels)
    

    f.close()


    #process end

    index = args.index

    data = imgs_database[index]
    label = labels_database[index]
    # data = imgs_database[index]
    # label = labels_database[index]
    print("-------------------------")
    print("label true is: ", cat[label])
    print("-------------------------")

    if args.dataset == "cifar10":
        img_orl = Image.fromarray(data).resize([224, 224], resample=Image.BILINEAR)####原来是224
    else :
        img_orl = Image.open(data).convert('RGB').resize([256, 256], resample=Image.BILINEAR)
    
    img_orl2 = crop_center(img_orl, 224, 224)####### cifar10我改成了32，cub200原来是224

    img_orl2.save(f'vis/{args.cid}_origin.png')# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None)
    print("-------------------------")
    pp = torch.argmax(pred, dim=-1)
    print("predicted as: ", cat[pp])

    w = model.state_dict()["cls.weight"][label]
    w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)
    ccc = np.around(cpt.cpu().detach().numpy(), 4)
    # draw_bar(w_numpy, name)

    # print("--------weight---------")
    # print(w_numpy)

    # print("--------cpt---------")
    # print(ccc)
    print("------sum--------")
    print((ccc/2 + 0.5) * w_numpy)
    if args.use_weight:
        w[w < 0] = 0
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), w)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
        heatmap_on_image.save("vis/" +  f'{args.cid}_0_slot_mask_{id}.png')

    # get retrieval cases
    f1 = h5py.File(f"data_map/{args.cid}_{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", 'r')
    database_hash = f1["database_hash"]
    database_labels = f1["database_labels"]
    test_hash = f1["test_hash"]
    test_labels = f1["test_labels"]

   
    print("-------------------------")
    print("generating concept samples")

    for j in range(args.num_cpt):
        root = 'vis_pp/' +str(args.cid)+  "cpt" + str(j) + "/"
        os.makedirs(root, exist_ok=True)
        selected = np.array(database_hash)[:, j]
        ids = np.argsort(-selected, axis=0)
        idx = ids[:args.top_samples]
        for i in range(len(idx)):
            current_is = idx[i]
            category = cat[int(database_labels[current_is][0])]
            #imagenet cub200
            if args.dataset == "MNIST":
                img_orl = Image.fromarray(imgs_database[current_is].numpy())
            elif args.dataset == "cifar10":
                img_orl = Image.fromarray(imgs_database[current_is])
            else:
                img_orl = Image.open(imgs_database[current_is]).convert('RGB')
            img_orl = img_orl.resize([256, 256], resample=Image.BILINEAR)
            img_orl2 = crop_center(img_orl, 224, 224)
            cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category, j])
            img_orl2.save(root + f'/orl_{i}_{category}.png')
            slot_image = np.array(Image.open(root + f'mask_{i}_{category}.png'), dtype=np.uint8)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
            heatmap_on_image.save(root + f'jet_{i}_{category}.png')

    print("end")

    model = MainModel(args,vis=True)
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    image = Image.open('test.jpg')

    transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    # 图片需要经过一系列数据增强手段以及统计信息(符合ImageNet数据集统计信息)的调整，才能输入模型
    image = transforms(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image=image.to(device)
    cpt1, pred1, att1, update1 = model(image)
    print("1")
    feature_output1 = model.featuremap1.transpose(1,0).cpu() 
    print("2")   
    feature_out1 = torchvision.utils.make_grid(feature_output1)
    print("3") 
    feature_imshow(feature_out1, 'feature_out1')                                             
  