import torch
import torch.nn.functional as F
from model.retrieval.loss import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
from .record import AverageMeter, ProgressMeter, show
from .tools import cal_acc, predict_hash_code, mean_average_precision


def train(args, model, device, loader, optimizer, epoch):
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
    progress = ProgressMeter(len(loader),
                             show_items,
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
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
            acc = cal_acc(cpt, label, False)
            pred_acces.update(acc)
            loss_total = retri_loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            progress.display(batch_idx)


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
def test(args, model, test_loader, device):
    model.eval()
    record = 0.0
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        cpt = model(data)
        retri_loss = F.nll_loss(cpt, label)
        acc = cal_acc(cpt, label, False)
        record += acc
    print("ACC:", record/len(test_loader))