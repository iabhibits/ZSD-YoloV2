import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import sparse

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def create_adjacencyList(data):
    A = np.zeros((20,20))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >= 0.65:
                A[i][j] = data[i][j]
    return A

def load_data(path='./graph_new.csv',feature_path='./glove_feature', dataset="VOC"):
    """Load citation network dataset (cora only for now)"""
    print('Loading coco_sim dataset...')


    #sim = pd.read_csv('./Similarity.csv',sep='\t')
    #data = sim.as_matrix()  
    #data = data[0:,2:]
    data = np.loadtxt(path,delimiter=",")
    features = np.loadtxt(feature_path)

    A = create_adjacencyList(data)
    row, col = np.where(A)
    coo = np.rec.fromarrays([row, col, A[row, col]], names='row col value'.split())
    adj = sparse.coo_matrix((coo['value'], (coo['row'], coo['col'])), (20, 20))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = sp.csr_matrix(features, dtype=np.float32)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
