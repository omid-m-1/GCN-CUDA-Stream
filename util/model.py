import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib
spmmv = importlib.import_module("deep-codegen.pytorch_apis").spmmv

# GCN Model
class GCNModel(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, rowPtr, colInd, values, device):
        super(GCNModel, self).__init__()
        # Dense layers
        self.W1 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_features, n_hidden, requires_grad=True)))
        self.W2 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_hidden, n_hidden, requires_grad=True)))
        self.W3 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_hidden, n_classes, requires_grad=True)))

        # Dropout
        self.dropout = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.4)

        # Send adjacency matrix to device
        self.rowPtr = rowPtr.to(device)
        self.colInd = colInd.to(device)
        self.values = values.to(device)

        # Print flag
        self.print_stream = True

    def forward(self, x):
        # First GCN layer
        x = x.mm(self.W1)
        x = self.dropout(F.relu(spmmv(x, self.rowPtr, self.colInd, self.values, self.print_stream)))
        if self.print_stream:
            self.print_stream = False

        # Second GCN layer
        x = x.mm(self.W2)
        x = self.dropout2(F.relu(spmmv(x, self.rowPtr, self.colInd, self.values)))

        # Classification layer
        x = x.mm(self.W3)
        x = F.log_softmax(x, dim=1)
        return x


