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


def eval(args,t,features,label):
    for i in range(20):
        p = torch.dot(t,features[i])
        #print(i,p)
    pred = torch.argmax(p)
    if label == pred:
        f = 1
    else:
        f = 0
    return f


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--hidden', type=int, default=100,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--model_path", type=str, default="./gcn_model/model.bin",help="Directory path to save trained model")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.cuda='cuda'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features = load_data()

    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=20,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = torch.tensor(features).to('cuda')
        adj = adj.cuda()

    file = open('sample','rb')
    target = pickle.load(file)

    model.to(args.cuda)
    model.load_state_dict(torch.load(args.model_path))

    l1_wt = model.gc1.weight
    l2_wt = model.gc2.weight
    
    features_l1 = torch.mm(features,l1_wt)
    features_l2 = f = torch.mm(features_l1,l2_wt)
    cnt = 0
    for i in range(len(target)):
        tar = target[i][1]
        label = int(target[i][0])
        cnt += eval(args,tar,features_l2,label)

    print("Model accuracy with layer2 is {}\n".format(cnt))

    #cnt = 0
    #for i in range(len(target)):
        #tar = target[i][1]
        #label = int(target[i][0])
        #cnt += eval(args,tar,l1_wt,label)
        #print(cnt)

    #print("Model accuracy with layer1 is {}\n".format(cnt))

if __name__ == '__main__':
    main()



