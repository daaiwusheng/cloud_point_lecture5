import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PointNetDataset
from model import PointNet
import torch.nn.functional as F


SEED = 13
gpus = [0]
batch_size = 1
ckp_path = '/home/steven/code/cloud_point/pointnet/point_net_log/latest.pth'

def load_ckp(ckp_path, model):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  print("model load from %s" % ckp_path)

if __name__ == "__main__":
  torch.manual_seed(SEED)
  device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
  print("Loading test dataset...")
  test_data = PointNetDataset("/home/steven/code/cloud_point/pointnet/data/modelnet40_normal_resampled", train=1)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  model = PointNet().to(device=device)
  if ckp_path:
    load_ckp(ckp_path, model)
    model = model.to(device)
  
  model.eval()

  with torch.no_grad():
    accs = []
    gt_ys = []
    pred_ys = []
    for x, y in test_loader:
      x = x.to(device)
      y = y.to(device)

      # TODO: put x into network and get out
      out = model(x)
      # TODO: get pred_y from out
      pred_y = torch.argmax(F.softmax(out), dim=1)
      gt = torch.argmax(y, dim=1)
      # TODO: calculate acc from pred_y and gt
      correct_num = (pred_y == gt).sum().item()
      acc = correct_num / x.size()[0]
      # gt_ys = np.append(gt_ys, gt)
      # pred_ys = np.append(pred_ys, pred_y)

      accs.append(acc)

    print("final acc is: " + str(np.mean(accs)))

    # final acc is: 0.7625607779578606
