# Code for concept

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of cpt")       #1.创建解析器使用： argparse 的第一步是创建一个 ArgumentParser 对象。 ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。parser.add_argument('--dataset', type=str, default="CUB200")
'''ArgumentParser 对象:
description - 在参数帮助文档之前显示的文本（默认值：无）
'''
parser.add_argument('--dataset_dir', type=str, default="/home/coder/projects/cub200/botcl-fed/datacub")       #2.添加参数：给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的

parser.add_argument('--output_dir', type=str, default="saved_model")
# ========================= Model Configs 模型配置==========================
parser.add_argument('--num_classes', default=10, type=int, help='category for classification')
parser.add_argument('--num_cpt', default=20, type=int, help='number of the concept')
parser.add_argument('--base_model', default="resnet18", type=str)
parser.add_argument('--img_size', default=224, help='size for input image')
parser.add_argument('--pre_train', default=False, type=bool, help='whether pre-train the model')
parser.add_argument('--aug', default=True, type=bool,help='whether use augmentation')
parser.add_argument('--num_retrieval', default=20, help='number of the top retrieval images')
parser.add_argument('--cpt_activation', default="att", help='the type to form cpt activation, default att using attention')
parser.add_argument('--feature_size', default=7, help='size of the feature from backbone')
parser.add_argument('--process', default=False, help='whether process for h5py file')
parser.add_argument('--layer', default=1, help='layers for fc, default as one')
parser.add_argument('--dataset', default="cifar10", help='dataset')#数据集选择
# ========================= Training Configs 训练配置==========================
#mnist 参数配置
# parser.add_argument('--weak_supervision_bias', type=float, default=0.1, help='weight fot the weak supervision branch')
# parser.add_argument('--att_bias', type=float, default=0.01, help='used to prevent overflow, default as 0.1')#2.调大点或者调小点 0.01或者0.5
# parser.add_argument('--quantity_bias', type=float, default=0.01, help='force each concept to be binary')#3.调小点 0.01
# parser.add_argument('--distinctiveness_bias', type=float, default=0, help='refer to paper')#4.设置为0
# parser.add_argument('--consistence_bias', type=float, default=0, help='refer to paper')#4.设置为0
#CUB200 参数配置
parser.add_argument('--weak_supervision_bias', type=float, default=0.1, help='weight fot the weak supervision branch')
parser.add_argument('--att_bias', type=float, default=0.5, help='used to prevent overflow, default as 0.1')#2.调大点或者调小点 0.01或者0.5
parser.add_argument('--quantity_bias', type=float, default=0.05, help='force each concept to be binary')#3.调小点 0.01
parser.add_argument('--distinctiveness_bias', type=float, default=0.01, help='refer to paper')#4.设置为0
parser.add_argument('--consistence_bias', type=float, default=0.1, help='refer to paper')#4.设置为0
# ========================= Learning Configs 学习配置==========================
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--lr', default=0.0001, type=float) #1.调小学习率
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr_drop', default=30, type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
# ========================= Machine Configs 机器配置==========================
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
# ========================= Demo Configs 演示配置==========================
parser.add_argument('--index', default=3, type=int)
parser.add_argument('--use_weight', default=False, help='whether use fc weight for the generation of attention mask')
parser.add_argument('--top_samples', default=20, type=int, help='top n activated samples')
# parser.add_argument('--demo_cls', default="n01498041", type=str)
parser.add_argument('--fre', default=3, type=int, help='frequent of show results during training')
parser.add_argument('--deactivate', default=-1, type=int, help='the index of concept to be deativated')

#fed
parser.add_argument("--server_address",type=str,default="[::]:8080",help=f"gRPC server address (default: '[::]:8080')",)
parser.add_argument("--cid",type=int,metavar="N",help="ID of current client (default: 0)",)
parser.add_argument("--nb_clients",type=int,default=5,metavar="N",help="Total number of clients being launched (default: 2)",)
parser.add_argument("--train-batch-size",type=int,default=64,metavar="N",help="input batch size for training (default: 64)",)
parser.add_argument("--test-batch-size",type=int,default=1000,metavar="N",help="input batch size for testing (default: 1000)",)
parser.add_argument("--epochs",type=int,default=5,metavar="N",help="number of epochs to train (default: 14)",)

