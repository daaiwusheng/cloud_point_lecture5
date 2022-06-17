from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import numpy as np
import time
import os
import datetime
import torch.nn.functional as F

from dataset import PointNetDataset
from model import PointNet
from utility import *

'''
以下参数获得76%的准确率,在验证集上
SEED = 13
batch_size = 75
epochs = 500
decay_lr_factor = 0.90
decay_lr_every = 25
lr = 0.0001
gpus = [0]
global_step = 0
show_every = 10
val_every = 3
'''


SEED = 13
batch_size = 75
epochs = 500
decay_lr_factor = 0.90
decay_lr_every = 25
lr = 0.0001
gpus = [0]
global_step = 0
show_every = 10
val_every = 3
date = datetime.date.today()
save_dir = "/home/steven/code/cloud_point/pointnet/point_net_log"


def save_ckp(ckp_dir, model, optimizer, epoch, best_acc, date):
    os.makedirs(ckp_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    ckp_path = os.path.join(ckp_dir, f'date_{date}-epoch_{epoch}-maxacc_{best_acc:.3f}.pth')
    torch.save(state, ckp_path)
    torch.save(state, os.path.join(ckp_dir, f'latest.pth'))
    print('model saved to %s' % ckp_path)


def load_ckp(ckp_path, model, optimizer):
    state = torch.load(ckp_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print("model load from %s" % ckp_path)


def softXEnt(prediction, real_class):
    # TODO: return loss here
    return F.cross_entropy(prediction, real_class)


def get_eval_acc_results(model, data_loader, device):
    """
    ACC
    """
    seq_id = 0
    model.eval()

    distribution = np.zeros([5])
    confusion_matrix = np.zeros([5, 5])
    pred_ys = []
    gt_ys = []
    with torch.no_grad():
        accs = []
        for x, y in data_loader:
            x = move_to_device(x, device)
            y = move_to_device(y, device)

            # TODO: put x into network and get out
            out = F.softmax(model(x))
            # TODO: get pred_y from out
            pred_y = torch.argmax(out, dim=1)
            # gt = np.argmax(y.cpu().numpy(), axis=1)
            gt = torch.argmax(y, dim=1)
            # TODO: calculate acc from pred_y and gt
            correct_num = (pred_y == gt).sum().item()
            acc = correct_num / x.size()[0]
            # gt_ys = np.append(gt_ys, gt)
            # pred_ys = np.append(pred_ys, pred_y)
            # idx = gt
            # acc = acc.cpu()
            accs.append(acc)

        return np.mean(accs)


if __name__ == "__main__":
    dir_name = f'pointnet_lr_{lr}_batch_size_{batch_size}_epochs_{epochs}' \
               f'_decay_lr_factor_{decay_lr_factor}_decay_every_{decay_lr_every}_'
    dir_need = '/home/steven/code/cloud_point/pointnet/point_net_log/runs/' + dir_name + '/'
    writer = SummaryWriter(dir_need)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    print("Loading train dataset...")
    train_data = PointNetDataset("/home/steven/code/cloud_point/pointnet/data/modelnet40_normal_resampled", train=0)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print("Loading valid dataset...")
    val_data = PointNetDataset("/home/steven/code/cloud_point/pointnet/data/modelnet40_normal_resampled/", train=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    print("Set model and optimizer...")
    model = PointNet().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    best_acc = 0.0
    model.train()
    print("Start trainning...")
    for epoch in range(epochs):
        acc_loss = 0.0
        num_samples = 0
        start_tic = time.time()
        for x, y in train_loader:
            x = move_to_device(x, device)
            y = move_to_device(y, device)
            # TODO: set grad to zero
            optimizer.zero_grad()
            # TODO: put x into network and get out
            out = model(x)
            loss = softXEnt(out, y)
            loss = move_to_device(loss, device)
            # TODO: loss backward
            loss.backward()
            # TODO: update network's param
            optimizer.step()
            acc_loss += x.size()[0] * loss.item()
            num_samples += y.shape[0]
            global_step += 1
            # TODO: get pred_y from out
            pred_y = torch.argmax(F.softmax(out), dim=1)
            gt = torch.argmax(y, dim=1)
            # TODO: calculate acc from pred_y and gt
            correct_num = (pred_y == gt).sum().item()
            acc = correct_num / x.size()[0]
            # print('acc: ', acc)
            if (global_step + 1) % show_every == 0:
                # ...log the running loss
                writer.add_scalar('training loss', acc_loss / num_samples, global_step)
                writer.add_scalar('training acc', acc, global_step)
                # print( f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        scheduler.step()
        print(
            f"loss at epoch {epoch}={acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f} sec")

        if (epoch + 1) % val_every == 0:

            acc = get_eval_acc_results(model, val_loader, device)
            print("eval at epoch[" + str(epoch) + f"] acc[{acc:3f}]")
            writer.add_scalar('validing acc', acc, epoch)

            if acc > best_acc:
                best_acc = acc
                save_ckp(save_dir, model, optimizer, epoch, best_acc, date)
                example = move_to_device(torch.randn(1, 3, 10000), device)
                traced_script_module = torch.jit.trace(model, example)
                traced_script_module.save("/home/steven/code/cloud_point/pointnet/point_net_log/traced_model.pt")
