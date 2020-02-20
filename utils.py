import torch
import torch.nn as nn
import numpy as np
import random
import ipdb
import os
from tqdm import trange
from torch.autograd import grad
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

avoid_zero_div = 1e-12
force_positive_weight = -1e-6

def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_perfect_model(model, r, device):    
    W, w ,b = return_model_params(model)
    W_shape = W.shape

    perfect_W = torch.cat([torch.eye(W_shape[1]), torch.zeros([W_shape[0] - W_shape[1], W_shape[1]])], dim = 0)
    list(model.parameters())[0].data = perfect_W
    list(model.parameters())[1].data = torch.tensor(1.).view(1,1)
    list(model.parameters())[2].data = torch.tensor(-1.).view(1)
    
    good_alpha = percentage_good_alpha(model, r)
    if np.abs( good_alpha - 100.) > 1e-5:
        raise Exception("Failed to make a perfect model! Good alpha: {0:.2f}%".format(percentage_good_alpha(model, r)))
    else:
        print("Good alpha: {0:.2f}%".format(percentage_good_alpha(model, r)))

    return model.to(device)

def make_positive_wb(model):
#     list(model.parameters())[1].data.clamp_(max = force_positive_weight)
    w = list(model.parameters())[1].detach()
    b = list(model.parameters())[2].detach()

    if b > 0:
        list(model.parameters())[2].data = (-b).detach()
    
    if w < 0:
        list(model.parameters())[2].data = (-w).detach()

def make_sphere(num_samples, dim, r, device):
    z = torch.randn([int(num_samples), dim])
    z_norm = torch.norm(z.detach(), p = 2, dim = 1, keepdim = True)
    x = (z / z_norm.clamp(min = avoid_zero_div)).to(device)
    mask = torch.rand(int(num_samples)) > 0.5
    x[mask,:] = r * x[mask,:]
    y = (r * mask.int()).float().to(device)
    
    return x, y

def make_true_max(model, r, device):
    alpha,u,v = compute_alpha(model, True)
    
    d_1 = 1/(r**2) - alpha[alpha < 1/(r**2)]
    d_2 = alpha[alpha > 1] - 1
    
    if len(d_1) == 0 and len(d_2) == 0:
        # all alphas are in the right range, we generate a random datapoint
        x, y = make_sphere(num_samples = 1, dim = len(alpha), r = r, device = device)
        y = y.view([])
    elif len(d_1) == 0 or d_1.max() <= (bool(len(d_2)) and d_2.max()):
        # d_1 is empty: all bad_alpha are larger than upper bound
        # OR max alpha error comes from alphas larger than upper bound
        # use direction corresponds to max alpha(svalue) to construct outer sphere example.
        x = v[:,0].to(device)
        x = (r * x.detach() / torch.norm(x.detach(), p = 2)).view(1,len(alpha))
        y = torch.tensor(1., device = device)
    else: 
        # d_2 is empty: all bad_alpha are smaller than lower bound
        # OR max alpha error comes from alphas smaller than lower bound
        # use direction corresponds to min alpha(svalue) to construct inner sphere example.
        x = v[:,-1].to(device)
        x = (x.detach() / torch.norm(x.detach(), p = 2)).view(1,len(alpha))
        y = torch.tensor(0., device = device)

    return x, y

def return_model_params(model):
    W = list(model.parameters())[0]
    w = list(model.parameters())[1]
    b = list(model.parameters())[2]
    return W, w ,b

def compute_alpha(model, uv = False):
    W, w ,b = return_model_params(model)
    u,s,v = torch.svd(W, compute_uv = uv)
    alpha = (w * s**2 / (-b)).view(-1)
    return alpha, u, v

def percentage_good_alpha(model, r):
    alpha,_,_ = compute_alpha(model, False)
    total_alpha = alpha.numel()
    good_alpha = ((alpha >= 1/(r**2)) & (alpha <= 1)).sum().item()
    
    return good_alpha / total_alpha *100

def analytic_err(model, r):
    alpha,_,_ = compute_alpha(model, False)

    # error rate based on inner sphere:
    mu_inner = (alpha - 1).sum().item()
    sigma_inner = np.sqrt(2*((alpha - 1)**2).sum().item())
    err_inner = 1 - norm.cdf(-mu_inner/sigma_inner)
    
    # error rate based on outer sphere:
    mu_outer = (r**2 - alpha).sum().item()
    sigma_outer = np.sqrt(2*((r**2 - alpha)**2).sum().item())
    err_outer = 1 - norm.cdf(-mu_outer/sigma_outer)

#     err = (err_inner + err_outer) * 0.5
    err = err_inner
    return err

def plot_err_stats(log, log_scale):
    train_acc = log["train_acc"]
    train_loss = log["train_loss"]
    err_1 = log["err_1"]
    err_2 = log["err_2"]
    iteration = log["iteration"]
    len_err = len(err_1)
    err_freq = int(iteration/len_err)
    iteration_list = list(range(err_freq, (len_err+1) * err_freq, err_freq))

    fig = plt.figure(figsize = [30,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,4)
    fig.add_subplot(gs[0,0]).plot(iteration_list, err_1, "C1", linewidth=3.0, marker = "o")
    fig.add_subplot(gs[0,1]).plot(iteration_list, err_2, "C2", linewidth=3.0, marker = "o")
    fig.add_subplot(gs[0,2]).plot(iteration_list, train_acc, "C3", linewidth=3.0, marker = "o")
    fig.add_subplot(gs[0,3]).plot(iteration_list, train_loss, "C4",label = "loss", linewidth=3.0, marker = "o")

    fig.add_subplot(gs[0,0]).set_title("Analytical error rate", fontsize = 25)
    fig.add_subplot(gs[0,1]).set_title("Percentage of good alphas" , fontsize = 25)
    fig.add_subplot(gs[0,2]).set_title("Train accuracy", fontsize = 25)
    fig.add_subplot(gs[0,3]).set_title("Train loss" , fontsize = 25)
    fig.add_subplot(gs[0,0]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0,1]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0,2]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0,3]).set_xlabel("iterations", fontsize = 25)
    
    if log_scale == True:
        fig.add_subplot(gs[0,0]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0,1]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0,2]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0,3]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0,0]).set_xscale("log")
        fig.add_subplot(gs[0,1]).set_xscale("log")
        fig.add_subplot(gs[0,2]).set_xscale("log")
        fig.add_subplot(gs[0,3]).set_xscale("log")

    fig.add_subplot(gs[0,0]).grid()
    fig.add_subplot(gs[0,1]).grid()
    fig.add_subplot(gs[0,2]).grid()
    fig.add_subplot(gs[0,3]).grid()
    fig.add_subplot(gs[0,0]).tick_params(labelsize=20)
    fig.add_subplot(gs[0,1]).tick_params(labelsize=20)
    fig.add_subplot(gs[0,2]).tick_params(labelsize=20)
    fig.add_subplot(gs[0,3]).tick_params(labelsize=20)

    fig.tight_layout()
    
    return fig

def save_stats(model, r, loss_clean, batch_acc, batch_size, i, log):
    log["err_1"].append(analytic_err(model, r))
    log["err_2"].append(percentage_good_alpha(model, r))
    log["train_acc"].append(batch_acc)
    log["train_loss"].append(loss_clean.item() * batch_size)
    log["iteration"] = i

    return log
