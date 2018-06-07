from __future__ import print_function
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
import numpy as np
import time
import math
import load_data
from models.new import CancerNet, FacesNetFloat, FacesNet, MnistNet, DiabetesNet
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Binary')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
#parser.add_argument('--gpus', default=3,
#                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--bn', type=int, default=1)
parser.add_argument('--bin_input', type=int, default=1,
                    help='input binary')
parser.add_argument('--drop_rate', type=float, default=0.0,
                    help='how often to drop')
parser.add_argument('--dropc_keep_prob', type=float, default=None,
                    help='fraction of inputs to use when computing thresholds')
parser.add_argument('--n_agg', type=int, default=1,
                    help='Number of classifiers to average')
parser.add_argument('--data', type=str, default='cancer')  # 'diabetes')
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--hf', type=int, default=32)
parser.add_argument('--cs', type=int, default=10)
parser.add_argument('--ir', type=float, default=1)
parser.add_argument('--divs', type=int, default=5)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
the_dim = 0
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.data == 'cancer':
    if args.bin_input:
        cancer_tr = load_data.NPZDataset('data/cancer/cancer_train.npz', 2, 0,
                                         as_int=False, divisions=args.divs)
        cancer_te = load_data.NPZDataset('data/cancer/cancer_test.npz',  2, 1,
                mins=cancer_tr.mins,
                maxes=cancer_tr.maxes,
                inter=cancer_tr.inter,
                                         as_int=False, divisions=args.divs)
    else:
        cancer_tr = load_data.NPZDataset('data/cancer/cancer_train.npz', 1, 0,
                                         as_int=False, divisions=args.divs)
        cancer_te = load_data.NPZDataset('data/cancer/cancer_test.npz',  1, 1,
                mean=cancer_tr.mean,
                std=cancer_tr.std,
                                         as_int=False, divisions=args.divs)

    train_loader = torch.utils.data.DataLoader(cancer_tr,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(cancer_te,
                                              batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    the_dim = 30*args.divs
elif args.data == 'faces':
    tr = load_data.NPZDataset2D('data/faces/faces_train.npz', 1, 0, as_int=False)
    te = load_data.NPZDataset2D('data/faces/faces_test.npz', 1, 1,
                                mean=tr.mean, std=tr.std, as_int=False)
    train_loader = torch.utils.data.DataLoader(
        tr, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        te, batch_size=args.batch_size, shuffle=True, **kwargs)
elif args.data == 'diabetes':
    if args.bin_input:
        diabetes_tr = load_data.NPZDataset('data/diabetes/diabetes_train.npz', 3, 0, divisions=args.divs)
        diabetes_te = load_data.NPZDataset(
            'data/diabetes/diabetes_test.npz', 3, 1, 
                    mins= diabetes_tr.mins,
                    maxes=diabetes_tr.maxes,
                    inter=diabetes_tr.inter,
                                             as_int=True, divisions=args.divs)
    else:
        diabetes_tr = load_data.NPZDataset('data/diabetes/diabetes_train.npz', 1, 0,
                                         as_int=True, divisions=args.divs)
        diabetes_te = load_data.NPZDataset('data/diabetes/diabetes_test.npz',  1, 1,
            mean=diabetes_tr.mean,
            std =diabetes_tr.std,
                                         as_int=True, divisions=args.divs)
    train_loader = torch.utils.data.DataLoader(
        diabetes_tr, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        diabetes_te, batch_size=args.batch_size, shuffle=True, **kwargs)
    the_dim = 1680 + 8*args.divs
elif args.data == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False,
                       transform=transforms.Compose(
                           [transforms.ToTensor(), transforms.Normalize(
                               (0.1307,), (0.3081,))])),
        batch_size=1000, shuffle=True, **kwargs)
else:
    print('nothing else at the moment')
    pdb.set_trace()


if args.data == 'cancer':
    if args.bin_input:
        model = CancerNet(args.bn, args.layers, the_dim, args.bin_input)
    else:
        model = CancerNet(args.bn, args.layers, 30, args.bin_input)
elif args.data == 'faces':
    if args.bin_input:
        model = FacesNet(args.layers, args.hf, args.cs)
    else:
        model = FacesNetFloat(args.layers, args.hf, args.cs)
elif args.data == 'diabetes':
    if args.bin_input:
        model = DiabetesNet(args.layers, args.hf, the_dim, args.bin_input)
    else:
        model = DiabetesNet(args.layers, args.hf, 1688, args.bin_input)
elif args.data == 'mnist':
    model = MnistNet(args.ir, args.layers)
else:
    print('nope')
    pdb.set_trace()

if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))
if args.data == 'cancer' or args.data == 'faces':
    criterion = nn.BCELoss(size_average=True)
else:
    criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

drop = args.drop_rate
dropc_keep_prob = args.dropc_keep_prob
n_agg = args.n_agg
print('dataset=' + str(args.data))
print('bin_input=' + str(args.bin_input))
print('drop_rate=' + str(args.drop_rate))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def validate(data, target, model, correct):
    if args.bin_input:
        data = (data>0).type(torch.FloatTensor)*2-1
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    if args.bin_input:
        output = model(data, percent=args.drop_rate)
    else:
        output = model(data)
    if args.data == 'cancer' or args.data == 'faces':
        pred = (output.data >= 0.5).type(torch.cuda.LongTensor)
    else:
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    #print(torch.mean(pred.type(torch.FloatTensor)))
    #pdb.set_trace()
    correct += pred.eq(target.data.view_as(pred).type(pred.type())).cpu().sum()
    return correct

def train(epoch):
    model.train()
    correct = 0
    val_correct = 0
    start = time.time()
    print('starting')
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # then we're checking validation
        if args.data == 'mnist' and batch_idx >= 500:
            val_correct = validate(data, target, model, val_correct)
            continue
        if args.bin_input:
            data = (data>0).type(torch.FloatTensor)*2-1 # binarize data!
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        if args.bin_input:
            output = model(data, percent=args.drop_rate)
        else:
            output = model(data)
        loss = criterion(output, target)
        if args.data == 'cancer' or args.data == 'faces':
            if args.cuda:
                pred = (output.data >= 0.5).type(torch.cuda.LongTensor)
            else:
                pred = (output.data >= 0.5).type(torch.LongTensor)
        else:
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(
            target.data.view_as(pred).type(pred.type())).cpu().sum()
        #if args.data == 'mnist':
        #    if epoch%10==0:
        #        optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2.0#*0.1

        optimizer.zero_grad()
        loss.backward()

        # NOTE: the 'org' parameters are the real-valued parameters that are updated during training
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            ti = timeSince(start)
            print('time taken=' + str(ti))
            start = time.time() 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    if args.data == 'mnist':
        print('\nValid set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_correct, 10000.0,
            100. * val_correct / 10000.0))
    return  (correct / (len(train_loader.dataset)*1.0)), (val_correct / 10000.0)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    for data, target in test_loader:
        pred = torch.LongTensor(n_agg, data.size()[0], 1)
        if args.bin_input:
            data = (data>0).type(torch.FloatTensor)*2-1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        for i in range(n_agg):
            if args.bin_input:
                output = model(data, percent=args.drop_rate,
                               dropc_keep_prob=dropc_keep_prob)
            else:
                output = model(data)
            test_loss += criterion(output, target).data[0]  # sum up batch loss
            if args.data == 'cancer' or args.data == 'faces':
                if args.cuda:
                    pred[i] = (output.data >= 0.5).view_as(pred[i]).type(
                        torch.cuda.LongTensor)
                else:
                    pred[i].copy_((output.data >= 0.5).view_as(pred[i]).type(
                        torch.LongTensor))
            else:
                # get the index of the max log-probability
                pred[i] = output.data.max(1, keepdim=True)[1]
            # pred = output.data.max(1, keepdim=True)[1]
            # get the index of the max log-probability
            # print(torch.mean(pred.type(torch.FloatTensor)))
        pred2 = torch.LongTensor(data.size()[0], 1)
        for i in range(data.size()[0]):
            counts = {}
            for j in range(n_agg):
                if not(pred[j, i] in counts.keys()):
                    counts[pred[j, i]] = 1
                else:
                    counts[pred[j, i]] += 1
            argmax, valmax = -1, -1
            for k, v in counts.items():
                if v > valmax:
                    argmax, valmax = k, v
            pred2[i] = argmax

        correct += pred2.eq(
            target.data.view_as(pred2).type(pred2.type())).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (correct / (len(test_loader.dataset)*1.0))

# =-=-=-=-= #
# MAIN CODE #
# =-=-=-=-= #
if args.data == 'faces':
    rest = '_bi' + str(args.bin_input) + \
        '_nagg' + str(args.n_agg) + '_l' + str(args.layers) + '_hf' + \
        str(args.hf) + '_cs' + str(args.cs) + '_drc' + str(args.dropc_keep_prob) + '_drop' + str(args.drop_rate)
elif args.data == 'cancer':
    rest = '_bi' + str(args.bin_input) + \
        '_nagg' + str(args.n_agg) + '_l' + str(args.layers) + \
        '_bn' + str(args.bn) + '_drc' + str(args.dropc_keep_prob) + '_divs' + str(args.divs) + '_seed' + str(args.seed) + '_drop' + str(args.drop_rate)
elif args.data == 'diabetes':
    rest = '_bi' + str(args.bin_input) + \
        '_nagg' + str(args.n_agg) + '_l' + str(args.layers) + \
        '_bn' + str(args.bn) + '_drc' + str(args.dropc_keep_prob) + '_divs' + str(args.divs) + '_dim' + str(args.hf) + '_drop' + str(args.drop_rate)
elif args.data == 'mnist':
    rest = '_bi' + str(args.bin_input) + \
        '_nagg' + str(args.n_agg) + '_l' + str(args.layers) + \
        '_ir' + str(args.ir) + '_drc' + str(args.dropc_keep_prob) + '_drop' + str(args.drop_rate)
else:
    rest = '_'
save_path = 'results/acc_' + args.data + rest
model_path= 'results/model_' + args.data + rest + '.model'
train_acc = np.zeros((args.epochs,))
valid_acc = np.zeros((args.epochs,))
test_acc  = np.zeros((args.epochs,))
for epoch in range(1, args.epochs + 1):
    train_acc[epoch-1], valid_acc[epoch-1] = train(epoch)
    test_acc[epoch-1]  = test()
    np.savez_compressed(save_path, train_acc=train_acc, test_acc=test_acc, valid_acc=valid_acc)
if args.data == 'faces':
    file_path = args.data + '_bin' + str(args.bin_input) + '_'
else:
    file_path = args.data + '_'
f = open(file_path + 'test.txt','a')
f.write(rest + ': ' + str(test_acc[-1]) + '\n')
f.close()
f = open(file_path + 'train.txt','a')
f.write(rest + ': ' + str(train_acc[-1]) + '\n')
f.close()
save_checkpoint({'args': args,
   'model': model.state_dict(),
   'optimizer': optimizer.state_dict()}, model_path)
