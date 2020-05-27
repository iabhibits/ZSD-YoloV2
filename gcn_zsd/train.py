from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN
import os
import glob
import argparse
import pickle
import cv2
from src.utils import *
from src.yolo_net import Yolo
from test_voc_images import *
#from test_voc_images import train_gcn

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--output_dir", type=str, default="gcn_model/",help="Directory path to save trained model")

# arguments for yolo model
# parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
# parser.add_argument("--conf_threshold", type=float, default=0.35)
# parser.add_argument("--nms_threshold", type=float, default=0.5)
# parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
# parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_voc")
# parser.add_argument("--input", type=str, default="test_images")
# parser.add_argument("--output", type=str, default="test_images")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=20,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    #labels = labels.cuda()
    #idx_train = idx_train.cuda()
    #idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()

#yolo_model = load_model(args)
#output_yolo = train_gcn(args)
emb = np.loadtxt('class_embedding')
emb = torch.tensor(emb,dtype=torch.float)
emb = emb.cuda()
print("The shape of embedding is : {}\n".format(emb.size()))

# load classifier weights
file = open('sample','rb')
target = pickle.load(file)
l = len(target)
mask = [0 for i in range(20)]
mask = torch.tensor(mask,dtype=torch.float64)

def cal_loss(loss,label,mask):
    l = []
    for i in loss:
        l.append(sum(i))
    l = torch.tensor(l,dtype=torch.float64,requires_grad=True)
    mask[label] = 1
    _loss = l * mask
    return _loss
    

def train(epoch,total_loss):
    model.train()
    optimizer.zero_grad()
    loss = torch.nn.MSELoss(reduction='none')
    output = model(features, adj)
    #print("The size of output is {}\n".format(output.size()))
    #label = output_yolo(epoch)
    e = epoch%l
    label = int(target[e][0])
    pred = output[label]
    _target = torch.tensor(target[e][1]).to('cuda')
    _target = _target.unsqueeze(0)
    _target = _target.repeat(20,1)
    loss_train = loss(pred,_target)
    #print("loss_train shape is {}\n".format(loss_train.shape))
    loss_train = cal_loss(loss_train,label,mask)
    total_loss += loss_train.mean().item()
    loss_train.mean().backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #acc_val = accuracy(output[idx_val], labels[idx_val])
    if epoch%10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),'iteration: {:04d}'.format(e+1),
              'loss_train: {:.4f}'.format(total_loss))
        total_loss = 0
    return total_loss

def test():
    model.eval()
    output = model(features, adj)
    # Required : change loss functio
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
epochs = l * args.epochs
total_loss = 0
print("Total loss is :{}\n".format(total_loss))
for epoch in range(epochs):
    total_loss = train(epoch,total_loss)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    out_dir = os.path.join(args.output_dir, "model2.bin")
else :
    out_dir = os.path.join(args.output_dir, "model2.bin")
torch.save(model.state_dict(), out_dir)

# Testing
#test()
