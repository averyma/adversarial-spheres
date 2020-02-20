import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
from models import *
from src.sphere import *
from src.utils_general import seed_everything
from torch.autograd import grad
import numpy as np
import warnings
import argparse
import os
warnings.filterwarnings("ignore")

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--method", default = "clean")
parser.add_argument("--lambbda", default = 1.0, type = float)
parser.add_argument("--pgd_itr", default = 10, type = int)
# parser.add_argument("--norm_clip", nargs = "+", default = [0,10], type = float)
# parser.add_argument("--lr", default = 0.001, type = float)
parser.add_argument("--seed", default = 100, type = int)
parser.add_argument("--checkpoint_dir", default = "dir_name")
parser.add_argument("--checkpoint_freq", default = 10, type = int)
args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_name = "custom_checkpoint.pth"
checkpoint_location = os.path.join(checkpoint_dir, checkpoint_name)
checkpoint_freq = args.checkpoint_freq
checkpoint_info = {"dir": checkpoint_dir,
                    "name": checkpoint_name,
                    "location": checkpoint_location,
                    "freq": checkpoint_freq}

if args.method == "clean":
    _name = "clean_seed" + str(args.seed)
if args.method == "adv":
    _name = "adv_pgd" + str(args.pgd_itr) + "_seed" + str(args.seed)
if args.method == "true_max":
    _name = "true_max_seed" + str(args.seed) 
if args.method == "reg_1st":
    _name = "reg_1st_lambda" + str(args.lambbda) + "_seed" + str(args.seed) 
if args.method == "reg_2nd":
    _name = "reg_2nd_lambda" + str(args.lambbda) + "_seed" + str(args.seed) 
print(_name)
_dir_name = "exp_jan26/sphere/"
_dir_root = "./exp/"+ _dir_name +"/"+_name
_path_log = "./exp/"+ _dir_name +"/"+_name+"/log.txt"

if not os.path.isdir(_dir_root):
    os.mkdir(_dir_root)
if not os.path.exists(_path_log):
    f = open(_path_log,"w+")
    f.close()

with open("./exp/"+ _dir_name +"/"+_name+"/log.txt","a") as f:
    f.write("\n\n"+str(checkpoint_dir[-5:]))

seed_everything(args.seed)
        
pgd_itr = args.pgd_itr
dim = 500
r = 1.3
err_freq = 100
# total_samples = 1e6
total_samples = 50e6
total_iterations = 1e4
batch_size = 50
lambbda = args.lambbda
_lr_rate = 0.0001 

_model = quad1().to(_device)
_model.to(_device)
_opt = optim.Adam(_model.parameters(), lr = _lr_rate) 

if args.method == "clean":
    log = train_clean(dim, r, total_samples, batch_size, err_freq, _model, _opt, _device, checkpoint_info)
if args.method == "true_max":
    log = train_true_max(dim, r, total_iterations, err_freq, _model, _opt, _device, checkpoint_info)
elif args.method == "adv":
    log = train_adv(pgd_itr, dim, r, total_samples, batch_size, err_freq, _model, _opt, _device, checkpoint_info)
elif args.method == "reg_1st":
    log = train_reg_1st(lambbda, dim, r, total_samples, batch_size, err_freq, _model, _opt, _device, checkpoint_info)
elif args.method == "reg_2nd":
    log = train_reg_2nd(lambbda, dim, r, total_samples, batch_size, err_freq, _model, _opt, _device, checkpoint_info)

fig = plot_err_stats(log)
fig.savefig("./exp/"+ _dir_name +"/"+_name + "/result.png")
