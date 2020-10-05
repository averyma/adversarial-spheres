import time
import os
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

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

def plot_stats(log, log_scale):
    itr_list = log["acc/batch"][:, 1]

    fig = plt.figure(figsize=[38, 7])

    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 5)
    fig.add_subplot(gs[0, 0]).plot(itr_list, log["ana_err/avg"][:, 2], "C2", label = "avg_err", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 0]).plot(itr_list, log["ana_err/inner"][:, 2], "C3", label = "inner", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 0]).plot(itr_list, log["ana_err/outer"][:, 2], "C4", label = "outer", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 1]).plot(itr_list, log["alpha/good"][:, 2], "C2", label = r"$\alpha_i \in [1/r^2, 1]$", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 1]).plot(itr_list, log["alpha/inner"][:, 2], "C3", label = r"$\alpha_i > 1$", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 1]).plot(itr_list, log["alpha/outer"][:, 2], "C4", label = r"$\alpha_i < 1/r^2$", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 4]).plot(itr_list, log["acc/batch"][:, 2], "C1", label = "train", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 4]).plot(itr_list, log["acc/test"][:, 2], "C0", label = "test", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 3]).plot(itr_list, log["loss/batch"][:, 2], "C1", label = "train", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 3]).plot(itr_list, log["loss/test"][:, 2], "C0", label = "test", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 2]).plot(itr_list, log["err_in_alpha/worst"][:, 2], "C3", label = "worst", linewidth=3.0, marker = "")
    fig.add_subplot(gs[0, 2]).plot(itr_list, log["err_in_alpha/avg"][:, 2], "C4", label = "average", linewidth=3.0, marker = "")

    fig.add_subplot(gs[0, 0]).set_title("Analytical error rate", fontsize = 25)
    fig.add_subplot(gs[0, 1]).set_title(r"Percentage of $\alpha_i$ in different ranges" , fontsize = 25)
    fig.add_subplot(gs[0, 4]).set_title("Accuracy", fontsize = 25)
    fig.add_subplot(gs[0, 3]).set_title("Loss" , fontsize = 25)
    fig.add_subplot(gs[0, 2]).set_title(r"Distance from the incorrect $\alpha_i$"+ "\n to the acceptable region" , fontsize = 25)
    fig.add_subplot(gs[0, 0]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0, 1]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0, 4]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0, 3]).set_xlabel("iterations", fontsize = 25)
    fig.add_subplot(gs[0, 2]).set_xlabel("iterations", fontsize = 25)
    
    if log_scale == True:
        fig.add_subplot(gs[0, 0]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0, 1]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0, 4]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0, 3]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0, 2]).ticklabel_format(style='sci', axis='x', scilimits=(5,5))
        fig.add_subplot(gs[0, 0]).set_xscale("log")
        fig.add_subplot(gs[0, 1]).set_xscale("log")
        fig.add_subplot(gs[0, 4]).set_xscale("log")
        fig.add_subplot(gs[0, 3]).set_xscale("log")
        fig.add_subplot(gs[0, 2]).set_xscale("log")

    fig.add_subplot(gs[0, 0]).grid()
    fig.add_subplot(gs[0, 1]).grid(which="both")
    fig.add_subplot(gs[0, 4]).grid()
    fig.add_subplot(gs[0, 3]).grid()
    fig.add_subplot(gs[0, 2]).grid()
    fig.add_subplot(gs[0, 0]).tick_params(labelsize=20)
    fig.add_subplot(gs[0, 1]).tick_params(labelsize=20)
    fig.add_subplot(gs[0, 4]).tick_params(labelsize=20)
    fig.add_subplot(gs[0, 3]).tick_params(labelsize=20)
    fig.add_subplot(gs[0, 2]).tick_params(labelsize=20)
    
    fig.add_subplot(gs[0, 0]).legend(prop={"size": 20})
    fig.add_subplot(gs[0, 1]).legend(prop={"size": 20})
    fig.add_subplot(gs[0, 4]).legend(prop={"size": 20})
    fig.add_subplot(gs[0, 3]).legend(prop={"size": 20})
    fig.add_subplot(gs[0, 2]).legend(prop={"size": 20})

    fig.tight_layout()
    
    return fig

class metaLogger(object):
    def __init__(self, log_path, flush_sec=5):
        self.log_path = log_path
        self.log_dict = self.load_log(self.log_path)
        self.writer = SummaryWriter(log_dir=self.log_path, flush_secs=flush_sec)

    def load_log(self, log_path):
        try:
            log_dict = torch.load(log_path + "/log.pth.tar")
        except FileNotFoundError:
            log_dict = defaultdict(lambda: list())
        return log_dict

    def add_scalar(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        self.log_dict[name] += [(time.time(), int(step), float(val))]

    def add_scalars(self, name, val_dict, step):
        self.writer.add_scalars(name, val_dict, step)
        for key, val in val_dict.items():
            self.log_dict[name+key] += [(time.time(), int(step), float(val))]

    def add_figure(self, name, val, step):
        self.writer.add_figure(name, val, step)
        val.savefig(self.log_path + "/" + name + ".png")

    def save_log(self, filename="log.pth.tar"):
        try:
            os.makedirs(self.log_path)
        except os.error:
            pass
        torch.save(dict(self.log_dict), self.log_path+'/'+filename)

    # def log_obj(self, name, val):
        # self.logobj[name] = val

    # def log_objs(self, name, val, step=None):
        # self.logobj[name] += [(time.time(), step, val)]

    # def log_vector(self, name, val, step=None):
        # name += '_v'
        # if step is None:
            # step = len(self.logobj[name])
        # self.logobj[name] += [(time.time(), step, list(val.flatten()))]

    def close(self):
        self.writer.close()
