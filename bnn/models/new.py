import torch
import torch.nn as nn
import torch.nn.functional as F
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d,TernarizeLinear,TernarizeConv2d
import pdb

class CancerNet(nn.Module):
    def __init__(self, use_bn, layers, d, is_bin):
        super(CancerNet, self).__init__()
        self.use_bn = use_bn
        self.layers = layers
        self.is_bin = is_bin
        fc = []
        bn = []
        for i in range(layers-1):
            if is_bin:
                fc.append(TernarizeLinear(d, d)) #(30, 30))
            else:
                fc.append(nn.Linear(d, d)) #(30, 30))
            if use_bn:
                bn.append(nn.BatchNorm1d(d))
        if is_bin:
            fc.append(TernarizeLinear(d, 1))
        else:
            fc.append(nn.Linear(d, 1))
        if use_bn:
            bn.append(nn.BatchNorm1d(1))
        self.fclist = nn.ModuleList(fc)
        self.bnlist = nn.ModuleList(bn)
        self.ht = nn.Hardtanh()
        self.sigmoid=nn.Sigmoid()
    def forward(self, x, percent=None, dropc_keep_prob=None):
        for i in range(self.layers):
            if self.is_bin:
                x = self.fclist[i](x, percent, dropc_keep_prob=dropc_keep_prob)
            else:
                x = self.fclist[i](x)
            if self.use_bn:
                x = self.bnlist[i](x)
            if i != self.layers-1:
                if self.is_bin:
                    if self.training:
                        x = self.ht(x)
                    else:
                        x = (x >= 0).float()*2-1
                else:
                    x = self.ht(x)
        return self.sigmoid(x)




class FacesNet(nn.Module):
    def __init__(self, layers, hidden_filters, conv_size):
        super(FacesNet, self).__init__()
        self.layers = layers
        convlist = []
        bnlist   = []
        num_inputs = 50
        convlist.append(TernarizeConv2d(1, hidden_filters, conv_size))
        bnlist.append(nn.BatchNorm2d(hidden_filters))
        num_inputs = (num_inputs - conv_size) + 1
        for l in range(layers-2):
            convlist.append(TernarizeConv2d(hidden_filters, hidden_filters, conv_size))
            bnlist.append(nn.BatchNorm2d(hidden_filters))
            num_inputs = (num_inputs - conv_size) + 1
        convlist.append(TernarizeConv2d(hidden_filters, 1, conv_size))
        bnlist.append(nn.BatchNorm2d(1))
        num_inputs = (num_inputs - conv_size) + 1
        bnlist.append(nn.BatchNorm2d(1))
        self.convlist = nn.ModuleList(convlist)
        self.bnlist   = nn.ModuleList(bnlist)
        self.num_inputs = num_inputs
        self.linear = TernarizeLinear(num_inputs*num_inputs, 1)
        self.ht = nn.Hardtanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x, percent=None, dropc_keep_prob=None):
        for l in range(self.layers):
            x = self.convlist[l](x, percent, dropc_keep_prob)
            x = self.bnlist[l](x)
            if self.training:
                x = self.ht(x)
            else:
                x = (x >= 0).float()*2-1
        x = x.view(-1,self.num_inputs*self.num_inputs)
        x = self.linear(x, percent, dropc_keep_prob)
        x = self.bnlist[-1](x)
        return self.sigmoid(x)

class DiabetesNet(nn.Module):
    def __init__(self, layers, d, the_dim, is_bin):
        super(DiabetesNet, self).__init__()
        fclist = []
        bnlist = []
        self.layers = layers
        self.d = d
        self.is_bin = is_bin
        if is_bin:
            fclist.append(TernarizeLinear(the_dim, d))
        else:
            fclist.append(nn.Linear(the_dim, d))
        bnlist.append(nn.BatchNorm1d(d))
        for i in range(layers-2):
            if is_bin:
                fclist.append(TernarizeLinear(d, d))
            else:
                fclist.append(nn.Linear(d, d))
            bnlist.append(nn.BatchNorm1d(d))
        if is_bin:
            fclist.append(TernarizeLinear(d, 3))
        else:
            fclist.append(nn.Linear(d, 3))
        bnlist.append(nn.BatchNorm1d(3))
        self.fclist = nn.ModuleList(fclist)
        self.bnlist = nn.ModuleList(bnlist)
        self.ht = nn.Hardtanh()
        self.logsoftmax=nn.LogSoftmax()
    def forward(self, x, percent=None, dropc_keep_prob=None):
        for i in range(self.layers):
            if self.is_bin:
                x = self.fclist[i](x, percent, dropc_keep_prob)
            else:
                x = self.fclist[i](x)
            x = self.bnlist[i](x)
            if i != self.layers-1:
                if self.is_bin:
                    if self.training:
                        x = self.ht(x)
                    else:
                        x = (x >= 0).float()*2-1
                else:
                    x = self.ht(x)
        return self.logsoftmax(x)

class MnistNet(nn.Module):
    def __init__(self, infl_ratio, layers):
        super(MnistNet, self).__init__()
        self.infl_ratio=infl_ratio
        self.ht = nn.Hardtanh()
        self.layers = layers
        fclist = []
        bnlist = []
        fclist.append(TernarizeLinear(784, int(1024*self.infl_ratio)))
        bnlist.append(nn.BatchNorm1d(int(1024*self.infl_ratio)))
        for i in range(layers-2):
            fclist.append(TernarizeLinear(int(1024*self.infl_ratio), int(1024*self.infl_ratio)))
            bnlist.append(nn.BatchNorm1d(int(1024*self.infl_ratio)))
        fclist.append(TernarizeLinear(int(1024*self.infl_ratio), 10))
        bnlist.append(nn.BatchNorm1d(10))
        self.fclist = nn.ModuleList(fclist)
        self.bnlist = nn.ModuleList(bnlist)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x, percent=None, dropc_keep_prob=None):
        x = x.view(-1, 28*28)
        for i in range(self.layers):
            x = self.fclist[i](x, percent, dropc_keep_prob)
            x = self.bnlist[i](x)
            if i != self.layers-1:
                if self.training:
                    x = self.ht(x)
                else:
                    x = (x >= 0).float()*2-1
        return self.logsoftmax(x)


# face float model
class FacesNetFloat(nn.Module):
    def __init__(self, layers, hidden_filters, conv_size):
        super(FacesNetFloat, self).__init__()
        self.layers = layers
        convlist = []
        bnlist   = []
        num_inputs = 50
        convlist.append(nn.Conv2d(1, hidden_filters, conv_size))
        bnlist.append(nn.BatchNorm2d(hidden_filters))
        num_inputs = (num_inputs - conv_size) + 1
        for l in range(layers-2):
            convlist.append(nn.Conv2d(hidden_filters, hidden_filters, conv_size))
            bnlist.append(nn.BatchNorm2d(hidden_filters))
            num_inputs = (num_inputs - conv_size) + 1
        convlist.append(nn.Conv2d(hidden_filters, 1, conv_size))
        bnlist.append(nn.BatchNorm2d(1))
        num_inputs = (num_inputs - conv_size) + 1
        bnlist.append(nn.BatchNorm2d(1))
        self.convlist = nn.ModuleList(convlist)
        self.bnlist   = nn.ModuleList(bnlist)
        self.num_inputs = num_inputs
        self.linear = nn.Linear(num_inputs*num_inputs, 1)
        self.ht = nn.Hardtanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x, dropc_keep_prob=None):
        for l in range(self.layers):
            x = self.convlist[l](x)
            x = self.bnlist[l](x)
            x = self.ht(x)
        x = x.view(-1,self.num_inputs*self.num_inputs)
        x = self.linear(x)
        x = self.bnlist[-1](x)
        return self.sigmoid(x)
