from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torch
from termcolor import colored
from utils.engine_recon import train, evaluation, vis_one
import torch.nn as nn

from configs import parser
from model.reconstruct.model_main import ConceptAutoencoder
import os


os.makedirs('saved_model/', exist_ok=True)


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  #torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起：
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    '''
    train（bool，可选）–如果为True，则从training.pt创建数据集，否则从test.pt创建数据集; 
    下载（bool，可选）–如果为true，则从internet下载数据集并将其放在根目录中。如果数据集已下载，则不会再次下载。
    transform（可调用，可选）–接受PIL图像并返回已转换版本的函数/转换。E、 g，变换。随机裁剪
    '''
    valset = datasets.MNIST('../data', train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=False)
    valloader = DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)
    model = ConceptAutoencoder(args, num_concepts=args.num_cpt)
    reconstruction_loss = nn.MSELoss()  #MSE是mean squared error的缩写，即平均平方误差，简称均方误差
    params = [p for p in model.parameters() if p.requires_grad] #p需要梯度
    optimizer = torch.optim.AdamW(params, lr=args.lr)   #AdamW就是Adam优化器加上L2正则，来限制参数值不可太大，
    '''
    params (iterable) – 优化器作用的模型参数
    lr (float) – learning rate – 相当于是统一框架中的阿尔法
    '''
    device = torch.device("cuda:1")
    model.to(device)    #代表将模型加载到指定设备上
    record_res = []
    record_att = []
    accs = []

    for i in range(args.epoch):
        print(colored('Epoch %d/%d' % (i + 1, args.epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr     调整学习率
        if i == args.lr_drop:
            print("Adjusted learning rate to 0.00001")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1     #动态修改学习率
        train(args, model, device, trainloader, reconstruction_loss, optimizer, i)      #train
        res_loss, att_loss, acc = evaluation(model, device, valloader, reconstruction_loss)     #评估
        record_res.append(res_loss)     #append() 方法向列表末尾追加元素
        record_att.append(att_loss)
        accs.append(acc)
        if i % args.fre == 0:   #arg.fre????????????????????????????????????????
            vis_one(model, device, valloader, epoch=i, select_index=1)
        print("Reconstruction Loss: ", record_res)
        print("Acc: ", accs)
        torch.save(model.state_dict(), f"saved_model/mnist_model_cpt{args.num_cpt}.pt")     #保存模型


if __name__ == '__main__':
    args = parser.parse_args()      #那么parser中增加的属性内容都会在args实例中，使用即可。
    args.att_bias = 5       #attention_    bias????????????????????
    args.quantity_bias = 0.1        #量偏置
    args.distinctiveness_bias = 0       #独特性偏置
    args.consistence_bias = 0       #一致性偏置
    os.makedirs(args.output_dir + '/', exist_ok=True)       #os.makedirs() 方法用于递归创建目录
    main()
