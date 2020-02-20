from scipy.stats import norm
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch.autograd import grad
import matplotlib.pyplot as plt
from src.context import ctx_noparamgrad_and_eval, ctx_eval
from src.save_function import checkpoint_save
import ipdb
import os

avoid_zero_div = 1e-12
force_positive_weight = -1e-6

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
    reg = log["reg"]
    iteration = log["iteration"]
    len_err = len(err_1)
    err_freq = int(iteration/len_err)
    iteration_list = list(range(err_freq, (len_err+1) * err_freq, err_freq))

    fig = plt.figure(figsize = [30,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,4)
    fig.add_subplot(gs[0,0]).plot(iteration_list, err_1, "C1", linewidth=3.0, marker = "o")
#     fig.add_subplot(gs[0,0]).set_yscale("log")
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
    
    if reg:
        fig.add_subplot(gs[0,3]).set_title("loss and reg" , fontsize = 25)
        fig.add_subplot(gs[0,3]).tick_params(axis='y', labelcolor="C4")
        twin = fig.add_subplot(gs[0,3]).twinx()
        twin.plot(iteration_list, reg, "C5", label = "reg",  linewidth=3.0, marker = "o")
        twin.tick_params(labelsize=20)
        twin.tick_params(axis='y', labelcolor="C5")
        if log_scale == True:
            twin.set_xscale("log")

        fig.add_subplot(gs[0,3]).legend(prop={"size": 20}, loc = "upper left")
        twin.legend(prop={"size": 20}, loc = "upper right")

    fig.tight_layout()
    
    return fig

def save_stats(model, r, loss_clean, batch_acc, batch_size, i, reg, log):
    log["err_1"].append(analytic_err(model, r))
    log["err_2"].append(percentage_good_alpha(model, r))
    log["train_acc"].append(batch_acc)
    log["train_loss"].append(loss_clean.item() * batch_size)
    log["iteration"] = i+1

    if reg:
        log["reg"].append(reg)

    return log

def load_checkpoint_sphere(checkpoint_info, model, opt, log):
    checkpoint = torch.load(checkpoint_info["location"])
    model.load_state_dict(checkpoint["state_dict"])
    opt.load_state_dict(checkpoint["optimizer"])
    log["train_loss"] =  checkpoint["train_loss"]
    log["train_acc"] = checkpoint["train_acc"]
    log["err_1"] = checkpoint["err_1"]
    log["err_2"] = checkpoint["err_2"]
    log["iteration"] = checkpoint["iteration"]
    log["reg"] = checkpoint["reg"]
    return model, opt, log

def save_checkpoint_sphere(checkpoint_info, model, opt, log):
    checkpoint_save({"state_dict": model.state_dict(),
                      "optimizer": opt.state_dict(),
                      "train_loss": log["train_loss"],
                      "train_acc": log["train_acc"],
                      "err_1": log["err_1"],
                      "err_2": log["err_2"],
                      "reg": log["reg"],
                      "iteration": log["iteration"]+1}, checkpoint_info["dir"], checkpoint_info["name"])

def reg_sphere(X, y, loss, clamp_value, model, device):
    
#     ipdb.set_trace()
    r = torch.norm(X.detach(), p = 2, dim = 1, keepdim = True)
    X_z = torch.zeros_like(X.detach(), requires_grad=True)
    _dim = X.shape[1]
    _batch = X.shape[0]
    
    dldx = len(X) * list(grad(loss, X, create_graph=True))[0].view(-1,_dim)
    
    z_d = torch.randn_like(X.detach(), requires_grad = True, device = device)
    z = torch.randn(_batch, requires_grad = True, device = device)
    
    z_d_norm = torch.norm(z_d.detach(), p = 2, dim = 1, keepdim = True).clamp(min = avoid_zero_div)
    
    ### step size?
    h = torch.empty([_batch,1], device = device).uniform_(0, 0.001)

    z_d.data = z_d.detach() / z_d_norm
    X_z.data = X.detach() + h * z_d.detach()
    
    yp_z = model(X_z)
    loss_z = nn.BCEWithLogitsLoss()(yp_z, y)
    dldx_z = len(X_z) * list(grad(loss_z, X_z, create_graph = True))[0].view(-1,_dim)
    Hz =  z_d_norm.view(-1,1) * (dldx_z - dldx)/h.view(-1,1).clamp(min = avoid_zero_div)

    top = Hz.view(-1,_dim) + z.view(-1,1) * dldx
    bot = torch.matmul(dldx.view(-1,1,_dim), z_d.view(_batch, _dim, 1)).view(-1,1) + z.view(-1,1)
    
    H_tilde_z_tilde = torch.cat([top, bot], dim = 1)
    H_tilde_z_tilde_norm = torch.norm(H_tilde_z_tilde, p = 2, dim = 1)
    reg = H_tilde_z_tilde_norm.mean()
    
    return reg
#     H_tilde_z_tilde_norm_clamp_max = H_tilde_z_tilde_norm.clamp(max = clamp_value[1])
#     H_tilde_z_tilde_norm_clamp = H_tilde_z_tilde_norm_clamp_max.clamp(min = clamp_value[0])
    
#     H_tilde_z_tilde_norm_clamp_mean = H_tilde_z_tilde_norm_clamp.mean()
    
#     total_element = H_tilde_z_tilde_norm.numel()
#     clamped_by_max_min = torch.ne(H_tilde_z_tilde_norm, H_tilde_z_tilde_norm_clamp).sum().item() 
#     clamped_by_max = torch.ne(H_tilde_z_tilde_norm, H_tilde_z_tilde_norm_clamp_max).sum().item() 
#     clamped_by_min = torch.ne(H_tilde_z_tilde_norm_clamp_max, H_tilde_z_tilde_norm_clamp).sum().item() 
#     clamp_freq = clamped_by_max_min/total_element*100
#     clamp_freq_max = clamped_by_max/total_element*100
#     clamp_freq_min = clamped_by_min/total_element*100

#     H_tilde_z_tilde_norm_unclamp_mean = H_tilde_z_tilde_norm.detach().mean()

#     reg = [H_tilde_z_tilde_norm_clamp_mean, H_tilde_z_tilde_norm_unclamp_mean]
#     clamp_freq = [clamp_freq, clamp_freq_min, clamp_freq_max]
#     return reg, clamp_freq

###################################################################################################
###################################################################################################
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
                
###################################################################################################
###################################################################################################
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

###################################################################################################
###################################################################################################
def train_clean(dim, r, total_samples, batch_size, err_freq, model, opt, device, checkpoint_info):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0, "reg": []}
    if checkpoint_info and os.path.exists(checkpoint_info["location"]):
        model, opt, log = load_checkpoint_sphere(checkpoint_info, log)
    checkpoint_itr = log["iteration"]

    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(checkpoint_itr,num_itr) as t:
        for i in range(checkpoint_itr, num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            yp = model(x)
            
            loss_clean = nn.BCEWithLogitsLoss()(yp, y)

            opt.zero_grad()
            loss_clean.backward()
            opt.step()
            
            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            # ipdb.set_trace()
            batch_acc = batch_correct / batch_size * 100

            t.set_postfix(loss = loss_clean.item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
            if (i+1) % err_freq == 0:
#                 print(percentage_good_alpha(model, r))
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i, False, log)
            if checkpoint_info and ((i+1) % checkpoint_info["freq"] == 0):
                save_checkpoint_sphere(checkpoint_info, model, opt, log) 
                print("SAVED CHECKPOINT")
    return log

def train_adv(pgd_itr, dim, r, total_samples, batch_size, err_freq, model, opt, device, checkpoint_info):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0, "reg": []}
    if checkpoint_info and os.path.exists(checkpoint_info["location"]):
        model, opt, log = load_checkpoint_sphere(checkpoint_info, log)
    checkpoint_itr = log["iteration"]

    attack_param = {'alpha': 0.01, 'num_iter': pgd_itr, 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(checkpoint_itr,num_itr) as t:
        for i in range(checkpoint_itr, num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            with ctx_noparamgrad_and_eval(model):
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
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i, False, log)
                print(i+1, percentage_good_alpha(model,r))
            if checkpoint_info and  (i+1) % checkpoint_info["freq"] == 0:
                save_checkpoint_sphere(checkpoint_info, model, opt, log) 
                print("SAVED CHECKPOINT")
    return log

def train_true_max(dim, r, total_iterations, err_freq, model, opt, device, checkpoint_info):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0, "reg": []}
    if checkpoint_info and os.path.exists(checkpoint_info["location"]):
        model, opt, log = load_checkpoint_sphere(checkpoint_info, log)
    checkpoint_itr = log["iteration"]
    
    num_itr = int(total_iterations)
    model.train()
    with trange(checkpoint_itr, num_itr) as t:
        for i in range(checkpoint_itr, num_itr):

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
                log = save_stats(model, r, loss_clean, batch_acc, 1, i, False, log)
                print(i+1, percentage_good_alpha(model,r))
            if checkpoint_info and  (i+1) % checkpoint_info["freq"] == 0:
                save_checkpoint_sphere(checkpoint_info, model, opt, log) 
                print("SAVED CHECKPOINT")
    return log

def train_reg_1st(lambbda, dim, r, total_samples, batch_size, err_freq, model, opt, device, checkpoint_info):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0, "reg": []}
    if checkpoint_info and os.path.exists(checkpoint_info["location"]):
        model, opt, log = load_checkpoint_sphere(checkpoint_info, log)
    checkpoint_itr = log["iteration"]
    
    attack_param = {'ord': np.inf, 'alpha': 0.001, 'num_iter': 1, 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(checkpoint_itr,num_itr) as t:
        for i in range(checkpoint_itr, num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
            
#             with ctx_noparamgrad_and_eval(model):
#                 delta = pgd_sphere(**attack_param).generate(model,x,y)
                
#             x_delta = x.detach() + delta.detach()
            x.requires_grad = True
            yp = model(x)
            loss_clean = nn.BCEWithLogitsLoss()(yp, y)
#             ipdb.set_trace()
            dldx = len(x) * list(grad(loss_clean, x, create_graph=True))[0].view(-1,x.shape[1])
            reg = torch.norm(dldx, p = 2, dim = 1).mean()

            loss = loss_clean + lambbda * reg
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            

            t.set_postfix(loss = loss_clean.item(),
                          reg = reg.item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
            if (i+1) % err_freq == 0:
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i, reg.item(), log)
                print(i+1, percentage_good_alpha(model,r))
            if checkpoint_info and (i+1) % checkpoint_info["freq"] == 0:
                save_checkpoint_sphere(checkpoint_info, model, opt, log) 
                print("SAVED CHECKPOINT")
    return log

def train_reg_2nd(lambbda, dim, r, total_samples, batch_size, err_freq, model, opt, device, checkpoint_info):
    log = {"train_acc": [], "train_loss": [], "err_1": [], "err_2": [], "iteration": 0, "reg": []}
    if checkpoint_info and os.path.exists(checkpoint_info["location"]):
        model, opt, log = load_checkpoint_sphere(checkpoint_info, log)
    checkpoint_itr = log["iteration"]
    
    attack_param = {'ord': np.inf, 'alpha': 0.001, 'num_iter': 1, 'loss_fn': nn.BCEWithLogitsLoss()}
    num_itr = int(total_samples/batch_size)
    model.train()
    with trange(checkpoint_itr,num_itr) as t:
        for i in range(checkpoint_itr, num_itr):
            
            x, y = make_sphere(batch_size, dim, r, device)
#             with ctx_noparamgrad_and_eval(model):
#                 delta = pgd_sphere(**attack_param).generate(model,x,y)
                
#             x_delta = x.detach() + delta.detach()
            x.requires_grad = True
            yp = model(x)
            loss_clean = nn.BCEWithLogitsLoss()(yp, y)

#             reg, clamp_freq = reg_sphere(x, y, loss_clean, [0, np.inf], model, device)
            reg = reg_sphere(x, y, loss_clean, [0, np.inf], model, device)

            loss = loss_clean + lambbda*reg
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = ((yp>0).bool() == y.bool()).sum().item()
            batch_acc = batch_correct / batch_size * 100
            

            t.set_postfix(loss = loss_clean.item(),
                          reg = (lambbda*reg).item(),
                          acc = "{0:.2f}%".format(batch_acc))
            t.update()
            
            if (i+1) % err_freq == 0:
                log = save_stats(model, r, loss_clean, batch_acc, batch_size, i, reg.item(), log)
                print(i+1, percentage_good_alpha(model,r))
            if checkpoint_info and (i+1) % checkpoint_info["freq"] == 0:
                save_checkpoint_sphere(checkpoint_info, model, opt, log) 
                print("SAVED CHECKPOINT")
    return log
