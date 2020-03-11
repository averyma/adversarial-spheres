import torch
import torch.nn as nn
import numpy as np
import ipdb
import os
from tqdm import trange
from torch.autograd import grad
from utils_sphere import *
from utils_general import *
from attacks import pgd_sphere, pgd_sphere_normalized

AVOID_ZERO_DIV = 1e-12

def train_clean(dim, r, total_samples, batch_size, err_freq, model, opt, device):
    stats = {"acc": mty_array(2), "loss": mty_array(2), "ana_err": mty_array(3), "alpha": mty_array(3), "iteration": 0}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            yp = model(x)
            
            loss = nn.BCEWithLogitsLoss(reduction = "mean")(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100

            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc),
                          good_alpha = "nan" if stats["alpha"].size ==0 else "{0:.2f}%".format(stats["alpha"][-1,0]))
            t.update()
            
            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_percentage = get_alpha_percentage(model, r)
                stats = save_stats([loss.item(), test_loss], [batch_acc, test_acc], ana_err, alpha_percentage, i+1, stats)
                if np.abs( alpha_percentage[0] - 100.) < 1e-5:
                    break
    return stats 

def train_adv_mixed_normalized(param, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    stats = {"acc": mty_array(2), "loss": mty_array(2), "ana_err": mty_array(3), "alpha": mty_array(3), "iteration": 0}

    # attack_param = {'alpha': 0.01, 'num_iter': pgd_itr, 'loss_fn': nn.BCEWithLogitsLoss()}
    attack_param = {'alpha': param["alpha"], 'num_iter': param["num_iter"], 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere_normalized(**attack_param).generate(model,x,y)
            
            yp_clean = model(x)
            loss_clean = nn.BCEWithLogitsLoss()(yp_clean, y)
            yp_adv = model(x+delta)
            loss_adv = nn.BCEWithLogitsLoss()(yp_adv, y)
            
            loss = loss_clean + loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp_clean>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            
            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc),
                          good_alpha = "nan" if stats["alpha"].size ==0 else "{0:.2f}%".format(stats["alpha"][-1,0]))
            t.update()
            
            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_percentage = get_alpha_percentage(model, r)
                stats = save_stats([loss.item(), test_loss], [batch_acc, test_acc], ana_err, alpha_percentage, i+1, stats)
                if np.abs( alpha_percentage[0] - 100.) < 1e-5:
                    break
    return stats

def train_adv_mixed(param, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    stats = {"acc": mty_array(2), "loss": mty_array(2), "ana_err": mty_array(3), "alpha": mty_array(3), "iteration": 0}

    # attack_param = {'alpha': 0.01, 'num_iter': pgd_itr, 'loss_fn': nn.BCEWithLogitsLoss()}
    # attack_param = {'alpha': alpha, 'num_iter': 1, 'loss_fn': nn.BCEWithLogitsLoss()}
    attack_param = {'alpha': param["alpha"], 'num_iter': param["num_iter"], 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere(**attack_param).generate(model,x,y)
            
            yp_clean = model(x)
            loss_clean = nn.BCEWithLogitsLoss()(yp_clean, y)
            yp_adv = model(x+delta)
            loss_adv = nn.BCEWithLogitsLoss()(yp_adv, y)
            
            loss = loss_clean + loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp_clean>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            
            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc),
                          good_alpha = "nan" if stats["alpha"].size ==0 else "{0:.2f}%".format(stats["alpha"][-1,0]))
            t.update()
            
            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_percentage = get_alpha_percentage(model, r)
                stats = save_stats([loss.item(), test_loss], [batch_acc, test_acc], ana_err, alpha_percentage, i+1, stats)
                if np.abs( alpha_percentage[0] - 100.) < 1e-5:
                    break
    return stats

def train_adv(param, dim, r, total_samples, batch_size, err_freq, model, opt, device):
    stats = {"acc": mty_array(2), "loss": mty_array(2), "ana_err": mty_array(3), "alpha": mty_array(3), "iteration": 0}

    # attack_param = {'alpha': 0.01, 'num_iter': pgd_itr, 'loss_fn': nn.BCEWithLogitsLoss()}
    attack_param = {'alpha': param["alpha"], 'num_iter': param["num_iter"], 'loss_fn': nn.BCEWithLogitsLoss()}
    # attack_param = {'alpha': alpha, 'num_iter': 1, 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere(**attack_param).generate(model,x,y)
            
            yp = model(x+delta)
            loss = nn.BCEWithLogitsLoss()(yp, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            
            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc),
                          good_alpha = "nan" if stats["alpha"].size ==0 else "{0:.2f}%".format(stats["alpha"][-1,0]))
            t.update()
            
            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_percentage = get_alpha_percentage(model, r)
                stats = save_stats([loss.item(), test_loss], [batch_acc, test_acc], ana_err, alpha_percentage, i+1, stats)
                if np.abs( alpha_percentage[0] - 100.) < 1e-5:
                    break
    return stats 

def train_truemax(dim, r, total_iterations, err_freq, model, opt, device):
    stats = {"acc": mty_array(2), "loss": mty_array(2), "ana_err": mty_array(3), "alpha": mty_array(3), "iteration": 0}
    
    batch_size = 1
    num_itr = int(total_iterations)
    model.train()
    with trange(num_itr) as t:
        for i in range(num_itr):

            x, y = make_true_max(model, r, device)

            yp = model(x)
            loss = nn.BCEWithLogitsLoss(reduction = "mean")(yp, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
                        
            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct * 100
            
            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc),
                          sphere = "Inner" if y.item()== 0 else "Outer",
                          good_alpha = "nan" if stats["alpha"].size ==0 else "{0:.2f}%".format(stats["alpha"][-1,0]))

            t.update()
            
            if (i+1) % err_freq == 0:
                test_acc, test_loss = test_clean(dim, r, 10000, 500, model, device)
                ana_err = analytic_err(model, r)
                alpha_percentage = get_alpha_percentage(model, r)
                stats = save_stats([loss.item(), test_loss], [batch_acc, test_acc], ana_err, alpha_percentage, i+1, stats)
                if np.abs( alpha_percentage[0] - 100.) < 1e-5:
                    break
    return stats

def test_clean(dim, r, total_samples, batch_size, model, device):
    model = model.eval()
    total_loss, total_correct = 0.,0.
    
    num_itr = int(total_samples/batch_size)
    # with trange(num_itr) as t:
    for i in range(num_itr):
        x, y = make_sphere(batch_size, dim, r, device)

        yp = model(x)
        loss = nn.BCEWithLogitsLoss(reduction = "mean")(yp,y)
        # ipdb.set_trace() 
        total_correct += ((yp>0).bool() == y.bool()).sum().item()
        total_loss += loss.item() * batch_size
        # t.update()

    test_acc = total_correct / total_samples * 100
    test_loss = total_loss / total_samples
    
    return test_acc, test_loss
                
def test_adv(pgd_itr, dim, r, total_samples, batch_size, model, device):
    model = model.eval()
    total_loss, total_correct = 0.,0.

    attack_param = {'alpha': 0.01, 'num_iter': pgd_itr, 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    with trange(num_itr) as t:
        for i in range(num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            delta = pgd_sphere(**attack_param).generate(model,x,y)
            
            yp = model(x+delta)
            loss = nn.BCEWithLogitsLoss()(yp, y)

            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            total_loss += loss.item() * batch_size
            
            t.set_postfix(loss = loss.item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
    test_acc = total_correct / total_samples
    test_loss = total_loss / total_samples
    return test_acc, test_loss


