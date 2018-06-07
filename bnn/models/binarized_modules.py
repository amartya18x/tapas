import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
import pdb


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return (tensor >= 0).float()*2-1
        #return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size())).clamp_(0,1).mul_(2).add_(-1)


def Ternarize(tensor,percent):
    if percent == 0.0:
        result = (tensor >= 0).float()*2-1
        #result = tensor.sign()
        return result, 0
    flat_tensor  = tensor.view(-1)
    size = max(int(tensor.nelement() * (1 - percent) - 1), 0)#max(params.nelement() - 10, 0))
    # commented out below is slower by a second :)
    ##if size > tensor.nelement()/2.0:
    ##    BB, _ = torch.topk(torch.abs(flat_tensor), tensor.nelement()-size, largest=True)
    ##    elem = BB[-1]
    ##else:
    ##AA, _ = torch.topk(torch.abs(flat_tensor), size, largest=False)
    ##elem = AA[-1]
    ##mask_tensor   = (torch.abs(tensor) <= elem).type(tensor.type())
    sorted_tensor, _ = torch.sort(torch.abs(flat_tensor)) # sorted in ascending order
    mask_tensor   = (torch.abs(tensor) <= sorted_tensor[size]).type(tensor.type())
    result = ((tensor >= 0).float()*2-1 * mask_tensor)
    return result, size


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

import torch.nn._functions as tnnf


class TernarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        self.size = 0
    # mask is masking layer
    def forward(self, input, percent, dropc_keep_prob=None): #, mask):#drop):
        #if input.size(1) != 784:
        input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data, size = Ternarize(self.weight.org, percent)
        self.size = size
        # need to use a mask so can drop SAME weights for binary weights AND non-binary weights!!
        #mask = torch.Tensor(self.weight.data.size()).uniform_(0, 1)
        #mask = (mask >= drop).type(self.weight.data.type())
        #self.weight.data=nn.functional.dropout(self.weight.data, p=drop, training=True)*drop
        #self.weight.org =nn.functional.dropout(self.weight.org
        #self.weight.data = self.weight.data * mask
        #self.weight.org  = self.weight.org  * mask
        # Added by Varun
        if dropc_keep_prob is None:
            out = nn.functional.linear(input, self.weight)
        else:
            mask = torch.Tensor(self.weight.data.size()).uniform_(0, 1)
            mask = (mask >= (1 - dropc_keep_prob)).type(self.weight.data.type())
            # Horrible code
            # tmp_tensor = self.weight.data.clone()
            # self.weight.data = self.weight.data * mask
            out = nn.functional.linear(input, self.weight * Variable(mask)) / dropc_keep_prob
            # self.weight.data = tmp_tensor
        # commented out by Matt!!
        #if not self.bias is None:
        #    self.bias.org=self.bias.data.clone()
        #    out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
    # mask is masking layer
    def forward(self, input): #, mask):#drop):
        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        # need to use a mask so can drop SAME weights for binary weights AND non-binary weights!!
        #mask = torch.Tensor(self.weight.data.size()).uniform_(0, 1)
        #mask = (mask >= drop).type(self.weight.data.type())
        #self.weight.data=nn.functional.dropout(self.weight.data, p=drop, training=True)*drop
        #self.weight.org =nn.functional.dropout(self.weight.org
        #self.weight.data = self.weight.data * mask
        #self.weight.org  = self.weight.org  * mask
        out = nn.functional.linear(input, self.weight)
        ## commented out by Matt!!
        #if not self.bias is None:
        #    self.bias.org=self.bias.data.clone()
        #    out += self.bias.view(1, -1).expand_as(out)

        return out


class TernarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input, percent, dropc_keep_prob=None):
        #if input.size(1) != 3:
        input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data, size=Ternarize(self.weight.org, percent)
        # Added by Varun
        if dropc_keep_prob is None:
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups)
        else:
            mask = torch.Tensor(self.weight.data.size()).uniform_(0, 1)
            mask = (mask >= (1 - dropc_keep_prob)).type(self.weight.data.type())
            # Horrible code
            # tmp_tensor = self.weight.data.clone()
            # self.weight.data = self.weight.data * mask
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups) / dropc_keep_prob


        #if not self.bias is None:
        #    self.bias.org=self.bias.data.clone()
        #    out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        #if input.size(1) != 3:
        input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        #if not self.bias is None:
        #    self.bias.org=self.bias.data.clone()
        #    out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
