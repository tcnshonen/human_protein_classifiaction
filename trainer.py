import torch
import torch.nn.functional as F
from time import perf_counter
from config import *


def train(epoch, net, opt, loader):
    net.train()
    length = len(loader)
    t0 = perf_counter()
    
    for i, (data, target) in enumerate(loader):
        data, target = data.to(config.device), target.to(config.device, dtype=torch.long)
        opt.zero_grad()
        pred = net(data)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        opt.step()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} - {}s'.format(epoch, i+1, length, 100.*(i+1)/length, loss.item(), int(perf_counter() - t0)), end='\r')
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    
def test(net, loader):
    net.eval()
    length = len(loader)
    loss = 0
    correct = 0
    total = 0
    
    for i, (data, target) in enumerate(loader):
        data, target = data.to(config.device), target.to(config.device, dtype=torch.long)
        pred = net(data)
        loss += F.cross_entropy(pred, target).item()
        #_, pred_index = pred.max(1)
        #_, target_index = target.max(1)
        correct += sum(pred.max(1)[1] == target).item()
        total += len(target)
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    loss /= length
    print('\nLoss: {:.4f}\t Accuray: {:.4f}\n'.format(loss, correct / total))
    