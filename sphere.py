import torch
import torch.nn as nn
import numpy as np
import ipdb
import os
from tqdm import trange
from torch.autograd import grad
from utils import *

avoid_zero_div = 1e-12
force_positive_weight = -1e-6

class pgd_sphere(object):
    """ projected gradient desscent, with random initialization within the ball """
    def __init__(self, **kwargs):
        # define default attack parameters here:
        self.param = {'alpha': 0.01,
                      'num_iter': 2,
                      'restarts': 1,
                      'loss_fn': nn.BCEWithLogitsLoss()}
        # parse thru the dictionary and modify user-specific params
        self.parse_param(**kwargs) 
        
    def generate(self, model, x, y):
        alpha = self.param['alpha']
        num_iter = self.param['num_iter']
        restarts = self.param['restarts']
        loss_fn = self.param['loss_fn']
        
        _dim = x.shape[1]
        r = torch.norm(x.detach(), p = 2, dim = 1, keepdim = True)
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = r * (x.detach()+ delta.detach()) / torch.norm((x + delta).detach(), p = 2, dim = 1, keepdim = True) - x.detach()
        # delta.data = r * (x.data + delta.data) / torch.norm((x.data + delta.data), p = 2, dim = 1, keepdim = True) - x.data
        
        for t in range(num_iter):
            loss = loss_fn(model(x + delta), y)
            loss.backward()
            # first we need to make sure delta is within the specified lp ball
            delta_grad = delta.grad.detach()
            delta_grad_norm = torch.norm(delta_grad.detach(), p = 2 , dim = 1, keepdim = True).clamp(min = avoid_zero_div)
            delta.data = delta.detach() + alpha * delta_grad.detach() / delta_grad_norm.detach()
            delta.data = r * (x.detach()+ delta.detach()) / torch.norm((x + delta).detach(), p = 2, dim = 1, keepdim = True) - x.detach()
            delta.grad.zero_()

        return delta.detach()

    def parse_param(self, **kwargs):
        for key,value in kwargs.items():
            if key in self.param:
                self.param[key] = value
                
def train_clean(dim, r, total_samples, batch_size, err_freq, model, opt, device):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0}

    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            yp = model(x)
            
            loss_clean = nn.BCEWithLogitsLoss()(yp, y)

            opt.zero_grad()
            loss_clean.backward()
            opt.step()
            
            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100

            t.set_postfix(loss = loss_clean.item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
            if (i+1) % err_freq == 0:
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i+1, log)
    return log

def train_adv(pgd_itr, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0}

    attack_param = {'alpha': 0.01, 'num_iter': pgd_itr, 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere(**attack_param).generate(model,x,y)
            
            yp = model(x+delta)
            loss_clean = nn.BCEWithLogitsLoss()(yp, y)
            
            opt.zero_grad()
            loss_clean.backward()
            opt.step()

            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            
            t.set_postfix(loss = loss_clean.item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
            if (i+1) % err_freq == 0:
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i+1, log)
    return log

def train_true_max(dim, r, total_iterations, err_freq, model, opt, device):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0}
    
    batch_size = 1
    num_itr = int(total_iterations)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):

            x, y = make_true_max(model, r, device)

            yp = model(x)
            loss_clean = nn.BCEWithLogitsLoss()(yp, y)
            
            opt.zero_grad()
            loss_clean.backward()
            opt.step()
                        
            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / x.shape[0] * 100
            
            t.set_postfix(loss = loss_clean.item(),
                          acc = "{0:.2f}%".format(batch_acc),
                          sphere = y.item())
            t.update()
            
            if (i+1) % err_freq == 0:
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i+1, log)
    return log

def test_clean_sphere(dim, r, total_samples, batch_size, model, device):
    model = model.eval()
    total_loss, total_correct = 0.,0.
    
    num_itr = int(total_samples/batch_size)
    with trange(num_itr) as t:
        for i in range(num_itr):
            x, y = make_sphere(batch_size, dim, r, device)

            yp = model(x)
            loss = nn.BCEWithLogitsLoss()(yp,y.float())

            total_correct += ((yp.squeeze()>0).float() == y.squeeze().float()).sum().item()
            total_loss += loss.item() * batch_size
            t.update()
    
    test_acc = total_correct / total_samples
    test_loss = total_loss / total_samples
    
    return test_acc, test_loss
                
def test_adv_sphere(loader, model, attack, param, device):
    model = model.eval()
    total_loss, total_correct = 0.,0.
    with trange(len(loader)) as t:
        for X,y in loader:
            X,y = X.to(device), y.float().to(device)
            delta = attack(**param).generate(model,X,y.float())
            yp = model(X+delta)
            loss = nn.BCEWithLogitsLoss()(yp,y)

            total_correct += ((yp.squeeze()>0).float() == y.squeeze().float()).sum().item()
            total_loss += loss.item() * X.shape[0]
            t.update()
    return total_correct / len(loader.dataset), total_loss / len(loader.dataset)
