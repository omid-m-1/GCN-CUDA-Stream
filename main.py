import dgl
import argparse
import torch
import scipy.sparse as sp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from util.util import train_model, evaluate_model
from util.model import GCNModel

import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='downaload DGL graph data for CSCI 780-02')
  parser.add_argument('--dataset',type=str,default='cora')
  parser.add_argument('--n_hidden', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=10000)
  parser.add_argument('--learning_rate', type=float, default=5e-5)
  parser.add_argument('--mixed', default='float32', help='valid options: [float32, mixed_tc, mixed_amp]')
  args = parser.parse_args()

  # Hyperparameters
  n_hidden = args.n_hidden
  epochs = args.epochs
  learning_rate = args.learning_rate
  mixed = args.mixed

  # Set random seed
  seed_value = 42
  os.environ['PYTHONHASHSEED'] = str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

  # Check available device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if (mixed == 'mixed_tc'):
    # Set tensor cores for mixed percission
    torch.set_float32_matmul_precision('medium')

  # Load graph data
  if args.dataset == 'cora':
    data = dgl.data.CoraGraphDataset()
  elif args.dataset == 'citeseer':
    data = dgl.data.CiteseerGraphDataset()
  elif args.dataset == 'pubmed':
    data = dgl.data.PubmedGraphDataset()
  elif args.dataset == 'reddit':
    data = dgl.data.RedditDataset()
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  g = data[0]

  # add self loop and bidirected link
  g = dgl.remove_self_loop(g)
  g = dgl.to_bidirected(g, copy_ndata=True)
  g = dgl.add_self_loop(g)

  # Load adjacency matrix in csr
  col,row=g.edges(order='srcdst')
  torch.set_printoptions(edgeitems=100)
  numlist = torch.arange(col.size(0), dtype=torch.int32)
  adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
  row_ptr=torch.from_numpy(adj_csr.indptr)
  col_ind=torch.from_numpy(adj_csr.indices)
  values = g.edata['weight'] if 'weight' in g.edata else torch.ones(col_ind.shape[0], dtype=torch.float32)

  # Load node data
  features=g.ndata['feat']
  labels=g.ndata['label']
  n_features=features.shape[1]
  n_classes=data.num_labels if args.dataset != 'reddit' else data.num_classes
  train_mask = g.ndata['train_mask']
  test_mask = g.ndata['test_mask']

  # Define GCN model, loss function, and optimizer
  model = GCNModel(n_features, n_hidden, n_classes, row_ptr, col_ind, values, device).to(device)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Train and evaluate GCN model on non default stream
  stream = torch.cuda.Stream()
  # Use non default stream
  with torch.cuda.stream(stream):
    train_model(model, features, labels, train_mask, criterion, optimizer,
      epochs=epochs, device=device, mixed=(mixed=='mixed_amp'))
    evaluate_model(model, features, labels, test_mask, criterion,
      device=device, mixed=(mixed=='mixed_amp'))

