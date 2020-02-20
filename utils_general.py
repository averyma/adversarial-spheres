import torch
import numpy as np
import random
import ipdb
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from utils_sphere import *

AVOID_ZERO_DIV = 1e-12
FORCE_POSITIVE_WEIGHT = -1e-6

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
